import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 

def double_conv(in_channels, out_channels):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, 3, padding=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(out_channels, affine=False),
    nn.Conv2d(out_channels, out_channels, 3, padding=1),
    nn.ReLU(inplace=True)
  )   


class UNet(nn.Module):

  def __init__(self, in_ch, out_ch):
    super().__init__()
        
    self.dconv_down1 = double_conv(in_ch, 64)
    self.dconv_down2 = double_conv(64, 128)
    self.dconv_down3 = double_conv(128, 256)
    self.dconv_down4 = double_conv(256, 512)    

    self.maxpool = nn.MaxPool2d(2)
    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)    
    
    self.dconv_up3 = double_conv(256 + 512, 256)
    self.dconv_up2 = double_conv(128 + 256, 128)
    self.dconv_up1 = double_conv(128 + 64, 64)
    
    self.conv_last = nn.Conv2d(64, out_ch, 1)
    
    
  def forward(self, x):
    conv1 = self.dconv_down1(x)
    x = self.maxpool(conv1)

    conv2 = self.dconv_down2(x)
    x = self.maxpool(conv2)
    
    conv3 = self.dconv_down3(x)
    x = self.maxpool(conv3)   
    
    x = self.dconv_down4(x)
    
    x = self.upsample(x)    
    x = torch.cat([x, conv3], dim=1)
    
    x = self.dconv_up3(x)
    x = self.upsample(x)    
    x = torch.cat([x, conv2], dim=1)     

    x = self.dconv_up2(x)
    x = self.upsample(x)    
    x = torch.cat([x, conv1], dim=1)   
    
    x = self.dconv_up1(x)
    
    out = self.conv_last(x)
    
    return out


  
  

class HED(torch.nn.Module):
  def __init__(self):
    super(HED, self).__init__()

    self.netVggOne = torch.nn.Sequential(
      torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False),
      torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False)
    )

    self.netVggTwo = torch.nn.Sequential(
      torch.nn.MaxPool2d(kernel_size=2, stride=2),
      torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False),
      torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False)
    )

    self.netVggThr = torch.nn.Sequential(
      torch.nn.MaxPool2d(kernel_size=2, stride=2),
      torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False),
      torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False),
      torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False)
    )

    self.netVggFou = torch.nn.Sequential(
      torch.nn.MaxPool2d(kernel_size=2, stride=2),
      torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False),
      torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False),
      torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False)
    )

    #self.netVggFiv = torch.nn.Sequential(
    #  torch.nn.MaxPool2d(kernel_size=2, stride=2),
    #  torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
    #  torch.nn.ReLU(inplace=False),
    #  torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
    #  torch.nn.ReLU(inplace=False),
    #  torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
    #  torch.nn.ReLU(inplace=False)
    #)

    self.netScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
    self.netScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
    self.netScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
    self.netScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
    #self.netScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

    #self.netCombine = torch.nn.Sequential(
    #  torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
    #  torch.nn.Sigmoid()
    #)
    
    self.load_my_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-hed/network-' + 'bsds500' + '.pytorch', file_name='hed-' + 'bsds500').items() })

  def load_my_state_dict(self, state_dict):
 
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
             continue
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)

  def forward(self, tenInput):
    tenBlue = (tenInput[:, 0:1, :, :] * 255.0) - 104.00698793
    tenGreen = (tenInput[:, 1:2, :, :] * 255.0) - 116.66876762
    tenRed = (tenInput[:, 2:3, :, :] * 255.0) - 122.67891434

    tenInput = torch.cat([ tenBlue, tenGreen, tenRed ], 1)

    tenVggOne = self.netVggOne(tenInput)
    tenVggTwo = self.netVggTwo(tenVggOne)
    tenVggThr = self.netVggThr(tenVggTwo)
    tenVggFou = self.netVggFou(tenVggThr)
    #tenVggFiv = self.netVggFiv(tenVggFou)

    tenScoreOne = self.netScoreOne(tenVggOne)
    tenScoreTwo = self.netScoreTwo(tenVggTwo)
    tenScoreThr = self.netScoreThr(tenVggThr)
    tenScoreFou = self.netScoreFou(tenVggFou)
    #tenScoreFiv = self.netScoreFiv(tenVggFiv)

    # tenScoreOne = torch.nn.functional.interpolate(input=tenScoreOne, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
    # tenScoreTwo = torch.nn.functional.interpolate(input=tenScoreTwo, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
    # tenScoreThr = torch.nn.functional.interpolate(input=tenScoreThr, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
    # tenScoreFou = torch.nn.functional.interpolate(input=tenScoreFou, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
    # tenScoreFiv = torch.nn.functional.interpolate(input=tenScoreFiv, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)

    # return self.netCombine(torch.cat([ tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv ], 1))
    
    return F.sigmoid(tenScoreOne), F.sigmoid(tenScoreTwo), F.sigmoid(tenScoreThr), F.sigmoid(tenScoreFou)
    
class EGUNet(nn.Module):

  def __init__(self, in_ch, out_ch):
    super().__init__()
        
    self.dconv_down1 = double_conv(in_ch, 64)
    self.dconv_down2 = double_conv(64, 128)
    self.dconv_down3 = double_conv(128, 256)
    self.dconv_down4 = double_conv(256, 512)    

    
    self.maxpool = nn.MaxPool2d(2)
    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)    
    
    self.dconv_egx4 = double_conv(512 + 2, 512)
    
    self.dconv_up3 = double_conv(256 + 512 + 2, 256)
    self.dconv_up2 = double_conv(128 + 256 + 2, 128)
    self.dconv_up1 = double_conv(64 + 128 + 2, 128)
    
    self.dconv_final = double_conv(128, 64)
    self.conv_last = nn.Conv2d(64, out_ch, 1)
    
    self.edgenet = HED()
    for param in self.edgenet.parameters():
      param.requires_grad = False
    
  def forward(self, x):
    
    edge1, edge2, edge3, edge4 = self.edgenet(x[:, :3, :, :])
    tedge1, tedge2, tedge3, tedge4 = self.edgenet(torch.cat([ x[:,3:,:,:], x[:,3:,:,:], x[:,3:,:,:] ], 1))
    # [1, 64, 512, 512] [1, 128, 256, 256] [1, 256, 128, 128] [1, 512, 64, 64]
    
    # Down 1
    conv1 = self.dconv_down1(x) # [1, 512, 64, 64]
    x = self.maxpool(conv1)

    # Down 2
    conv2 = self.dconv_down2(x) # [1, 256, 128, 128]
    x = self.maxpool(conv2)
    
    # Down 3
    conv3 = self.dconv_down3(x) # [1, 128, 256, 256]
    x = self.maxpool(conv3)   
    
    # Down 4 (Final)
    x = self.dconv_down4(x) # [1, 64, 512, 512]
    
    # Edge Guide 4
    egx4 = torch.cat([x, edge4, tedge4], dim=1)
    x = self.dconv_egx4(egx4)

    # Up 4
    x = self.upsample(x) 
    x = torch.cat([x, conv3, edge3, tedge3], dim=1)
 
    # Up 3
    x = self.dconv_up3(x)
    x = self.upsample(x)   
    
    x = torch.cat([x, conv2, edge2, tedge2], dim=1)     

    # Up 2
    x = self.dconv_up2(x)
    x = self.upsample(x)    
    x = torch.cat([x, conv1, edge1, tedge1], dim=1)   
    
    # Up 1
    x = self.dconv_up1(x)
    x = self.dconv_final(x)
    out = self.conv_last(x)
    
    return [out, edge1, tedge1]
    
    
class RTFNet(nn.Module):

  def __init__(self, in_ch, n_class):
    super(RTFNet, self).__init__()

    self.num_resnet_layers = 152

    if self.num_resnet_layers == 18:
      resnet_raw_model1 = models.resnet18(pretrained=True)
      resnet_raw_model2 = models.resnet18(pretrained=True)
      self.inplanes = 512
    elif self.num_resnet_layers == 34:
      resnet_raw_model1 = models.resnet34(pretrained=True)
      resnet_raw_model2 = models.resnet34(pretrained=True)
      self.inplanes = 512
    elif self.num_resnet_layers == 50:
      resnet_raw_model1 = models.resnet50(pretrained=True)
      resnet_raw_model2 = models.resnet50(pretrained=True)
      self.inplanes = 2048
    elif self.num_resnet_layers == 101:
      resnet_raw_model1 = models.resnet101(pretrained=True)
      resnet_raw_model2 = models.resnet101(pretrained=True)
      self.inplanes = 2048
    elif self.num_resnet_layers == 152:
      resnet_raw_model1 = models.resnet152(pretrained=True)
      resnet_raw_model2 = models.resnet152(pretrained=True)
      self.inplanes = 2048

    ########  Thermal ENCODER  ########
 
    self.encoder_thermal_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) 
    self.encoder_thermal_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1), dim=1)
    self.encoder_thermal_bn1 = resnet_raw_model1.bn1
    self.encoder_thermal_relu = resnet_raw_model1.relu
    self.encoder_thermal_maxpool = resnet_raw_model1.maxpool
    self.encoder_thermal_layer1 = resnet_raw_model1.layer1
    self.encoder_thermal_layer2 = resnet_raw_model1.layer2
    self.encoder_thermal_layer3 = resnet_raw_model1.layer3
    self.encoder_thermal_layer4 = resnet_raw_model1.layer4

    ########  RGB ENCODER  ########
 
    self.encoder_rgb_conv1 = resnet_raw_model2.conv1
    self.encoder_rgb_bn1 = resnet_raw_model2.bn1
    self.encoder_rgb_relu = resnet_raw_model2.relu
    self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
    self.encoder_rgb_layer1 = resnet_raw_model2.layer1
    self.encoder_rgb_layer2 = resnet_raw_model2.layer2
    self.encoder_rgb_layer3 = resnet_raw_model2.layer3
    self.encoder_rgb_layer4 = resnet_raw_model2.layer4

    ########  DECODER  ########

    self.deconv1 = self._make_transpose_layer(TransBottleneck, self.inplanes//2, 2, stride=2) # using // for python 3.6
    self.deconv2 = self._make_transpose_layer(TransBottleneck, self.inplanes//2, 2, stride=2) # using // for python 3.6
    self.deconv3 = self._make_transpose_layer(TransBottleneck, self.inplanes//2, 2, stride=2) # using // for python 3.6
    self.deconv4 = self._make_transpose_layer(TransBottleneck, self.inplanes//2, 2, stride=2) # using // for python 3.6
    self.deconv5 = self._make_transpose_layer(TransBottleneck, n_class, 2, stride=2)
 
  def _make_transpose_layer(self, block, planes, blocks, stride=1, inplanes=None):

    upsample = None
    if stride != 1:
      upsample = nn.Sequential(
        nn.ConvTranspose2d(inplanes if inplanes else self.inplanes, planes, kernel_size=2, stride=stride, padding=0, bias=False),
        nn.BatchNorm2d(planes),
      ) 
    elif self.inplanes != planes:
      upsample = nn.Sequential(
        nn.Conv2d(inplanes if inplanes else self.inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
        nn.BatchNorm2d(planes),
      ) 
 
    for m in upsample.modules():
      if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight.data)
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

    layers = []

    for i in range(1, blocks):
      layers.append(block(inplanes if inplanes else self.inplanes, inplanes if inplanes else self.inplanes))

    layers.append(block(inplanes if inplanes else self.inplanes, planes, stride, upsample))
    self.inplanes = planes

    return nn.Sequential(*layers)
 
  def forward(self, input):

    rgb = input[:,:3]
    thermal = input[:,3:]

    verbose = False

    # encoder

    ######################################################################

    if verbose: print("rgb.size() original: ", rgb.size())  # (480, 640)
    if verbose: print("thermal.size() original: ", thermal.size()) # (480, 640)

    ######################################################################

    rgb = self.encoder_rgb_conv1(rgb)
    if verbose: print("rgb.size() after conv1: ", rgb.size()) # (240, 320)
    rgb = self.encoder_rgb_bn1(rgb)
    if verbose: print("rgb.size() after bn1: ", rgb.size())  # (240, 320)
    rgb = self.encoder_rgb_relu(rgb)
    if verbose: print("rgb.size() after relu: ", rgb.size())  # (240, 320)

    thermal = self.encoder_thermal_conv1(thermal)
    if verbose: print("thermal.size() after conv1: ", thermal.size()) # (240, 320)
    thermal = self.encoder_thermal_bn1(thermal)
    if verbose: print("thermal.size() after bn1: ", thermal.size()) # (240, 320)
    thermal = self.encoder_thermal_relu(thermal)
    if verbose: print("thermal.size() after relu: ", thermal.size())  # (240, 320)

    rgb = rgb + thermal

    rgb = self.encoder_rgb_maxpool(rgb)
    if verbose: print("rgb.size() after maxpool: ", rgb.size()) # (120, 160)

    thermal = self.encoder_thermal_maxpool(thermal)
    if verbose: print("thermal.size() after maxpool: ", thermal.size()) # (120, 160)

    ######################################################################

    rgb = self.encoder_rgb_layer1(rgb)
    if verbose: print("rgb.size() after layer1: ", rgb.size()) # (120, 160)
    thermal = self.encoder_thermal_layer1(thermal)
    if verbose: print("thermal.size() after layer1: ", thermal.size()) # (120, 160)

    rgb = rgb + thermal

    ######################################################################
 
    rgb = self.encoder_rgb_layer2(rgb)
    if verbose: print("rgb.size() after layer2: ", rgb.size()) # (60, 80)
    thermal = self.encoder_thermal_layer2(thermal)
    if verbose: print("thermal.size() after layer2: ", thermal.size()) # (60, 80)

    rgb = rgb + thermal

    ######################################################################

    rgb = self.encoder_rgb_layer3(rgb)
    if verbose: print("rgb.size() after layer3: ", rgb.size()) # (30, 40)
    thermal = self.encoder_thermal_layer3(thermal)
    if verbose: print("thermal.size() after layer3: ", thermal.size()) # (30, 40)

    rgb = rgb + thermal

    ######################################################################

    rgb = self.encoder_rgb_layer4(rgb)
    if verbose: print("rgb.size() after layer4: ", rgb.size()) # (15, 20)
    thermal = self.encoder_thermal_layer4(thermal)
    if verbose: print("thermal.size() after layer4: ", thermal.size()) # (15, 20)

    fuse = rgb + thermal

    ######################################################################

    # decoder

    fuse = self.deconv1(fuse)
    if verbose: print("fuse after deconv1: ", fuse.size()) # (30, 40)
    fuse = self.deconv2(fuse)
    if verbose: print("fuse after deconv2: ", fuse.size()) # (60, 80)
    fuse = self.deconv3(fuse)
    if verbose: print("fuse after deconv3: ", fuse.size()) # (120, 160)
    fuse = self.deconv4(fuse)
    if verbose: print("fuse after deconv4: ", fuse.size()) # (240, 320)
    fuse = self.deconv5(fuse)
    if verbose: print("fuse after deconv5: ", fuse.size()) # (480, 640)

    return fuse
  
class TransBottleneck(nn.Module):

  def __init__(self, inplanes, planes, stride=1, upsample=None):
    super(TransBottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)  
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)  
    self.bn2 = nn.BatchNorm2d(planes)

    if upsample is not None and stride != 1:
      self.conv3 = nn.ConvTranspose2d(planes, planes, kernel_size=2, stride=stride, padding=0, bias=False)  
    else:
      self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)  

    self.bn3 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.upsample = upsample
    self.stride = stride
 
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
      elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight.data)
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.upsample is not None:
      residual = self.upsample(x)

    out += residual
    out = self.relu(out)

    return out

def unit_test():
  num_minibatch = 2
  rgb = torch.randn(num_minibatch, 3, 480, 640).cuda(0)
  thermal = torch.randn(num_minibatch, 1, 480, 640).cuda(0)
  rtf_net = RTFNet(9).cuda(0)
  input = torch.cat((rgb, thermal), dim=1)
  rtf_net(input)
  #print('The model: ', rtf_net.modules)
  
if __name__ == '__main__':
  import pdb
  
  model = EGUNet(4,14)
  for name,param in model.named_parameters():
    print (name)
  x = torch.rand(1,4,512,512)
  out = model(x)
  