import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import distiller.modules

from .utils import (
    relu_fn,
    round_filters,
    round_repeats,
    drop_connect,
    Conv2dSamePadding,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    WSConv2d,
    WSConv2d_v1,
)

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params, enable_se=True, group_se = True, sampling = True, sampling_rate = 0.5, half_up_sampling=1, 
        group_first_1x1 =False, rm_first_1x1=False):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1) and enable_se
        self.id_skip = block_args.id_skip  # skip connection and drop connect
        self.rm_first_1x1 = rm_first_1x1
        self.add = distiller.modules.EltwiseAdd(inplace=True)
        multiplier1 = 1/(sampling_rate * half_up_sampling)
        multiplier2 = 1/sampling_rate
        # use learnable coefficeints to replace first 1x1 conv layer
        if self._block_args.expand_ratio > 1 and rm_first_1x1:
            self.coefficient = Parameter(torch.Tensor(self._block_args.expand_ratio), requires_grad=True)
            self.coeff_scale_a = Parameter(torch.Tensor(self._block_args.expand_ratio), requires_grad=True)
            self.coeff_scale_b = Parameter(torch.Tensor(self._block_args.expand_ratio), requires_grad=True)
        print("Using WSNetV2 conv", sampling)
        
        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        self.oup = oup
        if self._block_args.expand_ratio != 1 and (not rm_first_1x1):
            if group_first_1x1:
                self._expand_conv = Conv2dSamePadding(in_channels=inp, out_channels=oup, kernel_size=1, bias=False, groups=inp//4)
            else:
                if not sampling:
                    self._expand_conv = Conv2dSamePadding(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
                else:
                    self._expand_conv = WSConv2d_v1(inp, oup, kernel_size=1, bias=False, multiplier = multiplier1, rep_dim=0, use_coeff=False)
                    # print(self._expand_conv)
                    # import pdb; pdb.set_trace()
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2dSamePadding(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            if group_se:
                self._se_reduce = Conv2dSamePadding(in_channels=oup, out_channels=num_squeezed_channels, groups = num_squeezed_channels, kernel_size=1)
                self._se_expand = Conv2dSamePadding(in_channels=num_squeezed_channels, out_channels=oup, groups = num_squeezed_channels, kernel_size=1)
            else:
                self._se_reduce = Conv2dSamePadding(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
                self._se_expand = Conv2dSamePadding(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        if not sampling:
            self._project_conv = Conv2dSamePadding(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        else:
            self._project_conv = WSConv2d_v1(oup, final_oup, kernel_size=1, bias=False, multiplier = multiplier2, rep_dim=1, use_coeff=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
    def sampling_feature(self, x, output_dim):
        if self._block_args.expand_ratio == 1:
            return x
        x_tmp = torch.cat([x,x],dim=1)
        x_new = []
        # import pdb; pdb.set_trace()
        for i in range(self._block_args.expand_ratio):
            coeff = torch.sigmoid(self.coefficient[i]) * self._block_args.expand_ratio
            coeff_int = int(torch.ceil(coeff))
            coeff_frac = coeff - coeff_int
            x_new.append(x_tmp[:,coeff_int:int(coeff_int+x.shape[1]),:,:] * coeff_frac * self.coeff_scale_a + 
            x_tmp[:,int(coeff_int + 1):int(coeff_int + x.shape[1]+1),:,:] * (1-coeff_frac) * self.coeff_scale_b)
        
        return torch.cat(x_new,dim=1)
    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        
        if self._block_args.expand_ratio != 1:
            try:
                if self.rm_first_1x1:
                    x = relu_fn(self._bn0(self.sampling_feature(x, self.oup)))
                    # import pdb; pdb.set_trace()
                else:
                    x = relu_fn(self._bn0(self._expand_conv(inputs)))
            except:
                import pdb;pdb.set_trace()
        x = relu_fn(self._bn1(self._depthwise_conv(x)))
        
        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            try:
                x = torch.sigmoid(x_squeezed) * x
            except:
                import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            # x = x + inputs  # skip connection
            x = self.add(x,inputs)
        return x


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self, blocks_args=None, global_params=None, sampling=True, fc_compress = 'fully_fc'):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self.fc_compress = fc_compress
        self._blocks_args = blocks_args
        # indexing           1     2   3   4   5   6   7   8   9   10  11  12  13  14  15
        # uniform sampling rate
        self.sampling_cfg = [0.5, 0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
        # self.sampling_cfg = [0.5, 0.5,0.5,0.5,0.5,0.5,0.4,0.5,0.4,0.5,0.4,0.5,0.4,0.5,0.5]

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2dSamePadding(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        idx = 0
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            idx += 1
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params, sampling_rate = self.sampling_cfg[idx]))
                idx += 1
        self._feature_blocks = nn.Sequential(self._blocks)
        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        if sampling:
            self._conv_head = WSConv2d_v1(in_channels, out_channels, kernel_size=1, bias=False, multiplier = 4/2, rep_dim=1, use_coeff=False)
        else:
            self._conv_head = Conv2dSamePadding(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._dropout = self._global_params.dropout_rate
        if fc_compress == 'fact_fc':
            low_rank_channels = 320
            self._fc_r1 = nn.Linear(out_channels, low_rank_channels)
            self._fc_r2 = nn.Linear(low_rank_channels, self._global_params.num_classes)
        elif fc_compress == 'group_fc':
            self._fc = Conv2dSamePadding(out_channels, self._global_params.num_classes, kernel_size=1, groups=4, bias=True)
        else:
            self._fc = nn.Linear(out_channels, self._global_params.num_classes)

    def features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = relu_fn(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._feature_blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x) # , drop_connect_rate) # see https://github.com/tensorflow/tpu/issues/381
        # Head
        x = relu_fn(self._bn1(self._conv_head(x)))
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        if self._dropout:
            x = F.dropout(x, p=self._dropout, training=self.training)
        if self.fc_compress == 'fact_fc':
            x = self._fc_r1(x)
            x = self._fc_r2(relu_fn(self._fc_r1(x)))
        elif self.fc_compress == 'group_fc':
            x = self._fc(x.view(-1,1280,1,1))
        else:
            x = self._fc(x)
        return x

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """

        # Convolution layers
        x = self.features(inputs)

       # # Head
       # x = relu_fn(self._bn1(self._conv_head(x)))
       # x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
       # if self._dropout:
       #     x = F.dropout(x, p=self._dropout, training=self.training)
       # if self.fc_compress == 'fact_fc':
       #     x = self._fc_r1(x)
       #     x = self._fc_r2(relu_fn(self._fc_r1(x)))
       # elif self.fc_compress == 'group_fc':
       #     x = self._fc(x.view(-1,1280,1,1))
       # else:
       #     x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return EfficientNet(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name):
        model = EfficientNet.from_name(model_name)
        load_pretrained_weights(model, model_name)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        """ Validates model name. None that pretrained weights are only available for
        the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ['efficientnet_b'+str(i) for i in range(num_models)]
        if model_name.replace('-','_') not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))
