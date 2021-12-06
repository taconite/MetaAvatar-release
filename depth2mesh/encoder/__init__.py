from depth2mesh.encoder import pointnet

encoder_dict = {
    'pointnet_simple': pointnet.SimplePointnet,
    'pointnet_resnet': pointnet.ResnetPointnet,
    'pointnet_conv': pointnet.ConvPointnet,
}
