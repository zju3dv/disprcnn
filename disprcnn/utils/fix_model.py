import logging

from torch.nn.parallel import DistributedDataParallel


def fix_parameters(model, cfg):
    logger = logging.getLogger("disprcnn.trainer")
    fix_backbone, fix_rpn, fix_box_head, \
    fix_mask_head, fix_shape_head, fix_disparity, \
    fix_pointrcnn_rpn, fix_pointcloud = retreive_params(cfg)
    if fix_backbone and hasattr(model, 'backbone'):
        logger.info('fix backbone')
        for param in model.backbone.parameters():
            param.requires_grad = False
    if fix_rpn and hasattr(model, 'rpn'):
        logger.info('fix rpn')
        for param in model.rpn.parameters():
            param.requires_grad = False
    if fix_box_head and hasattr(model, 'roi_heads'):
        logger.info('fix box head')
        for param in model.roi_heads.box.parameters():
            param.requires_grad = False
    if fix_mask_head and hasattr(model, 'roi_heads'):
        logger.info('fix mask head')
        for param in model.roi_heads.mask.parameters():
            param.requires_grad = False
    if fix_shape_head and hasattr(model, 'roi_heads'):
        logger.info('fix shape head')
        for param in model.roi_heads.shape.parameters():
            param.requires_grad = False
    if fix_disparity and hasattr(model, 'dispnet'):
        for param in model.dispnet.parameters():
            param.requires_grad = False
    if fix_pointrcnn_rpn and hasattr(model, 'pcnet'):
        for param in model.pcnet.rpn.parameters():
            param.requires_grad = False
    if fix_pointcloud and hasattr(model, 'pcnet'):
        for param in model.pcnet.parameters():
            param.requires_grad = False
    return model


def fix_model_training(model, cfg):
    logger = logging.getLogger("disprcnn.trainer")
    fix_backbone, fix_rpn, fix_box_head, \
    fix_mask_head, fix_shape_head, fix_disparity, \
    fix_pointrcnn_rpn, fix_pointcloud = retreive_params(cfg)
    if not isinstance(model, DistributedDataParallel):
        if fix_backbone and hasattr(model, 'backbone'):
            logger.info('eval backbone')
            model.backbone.eval()
        if fix_rpn and hasattr(model, 'rpn'):
            logger.info('eval rpn')
            model.rpn.eval()
        if fix_box_head and hasattr(model, 'roi_heads'):
            logger.info('eval box head')
            model.roi_heads.box.eval()
        if fix_mask_head and hasattr(model, 'roi_heads'):
            logger.info('eval mask head')
            model.roi_heads.mask.eval()
        if fix_shape_head and hasattr(model, 'roi_heads'):
            logger.info('eval shape head')
            model.roi_heads.shape.eval()
        if fix_disparity and hasattr(model, 'dispnet'):
            logger.info('eval dispnet')
            model.dispnet.eval()
        if fix_pointrcnn_rpn and hasattr(model, 'pcnet'):
            logger.info('eval pointrcnn rpn')
            model.pcnet.rpn.eval()
        if fix_pointcloud and hasattr(model, 'pcnet'):
            logger.info('eval pcnet')
            model.pcnet.eval()
    else:
        if fix_backbone and hasattr(model.module, 'backbone'):
            logger.info('eval backbone')
            model.module.backbone.eval()
        if fix_rpn and hasattr(model.module, 'rpn'):
            logger.info('eval rpn')
            model.module.rpn.eval()
        if fix_box_head and hasattr(model.module, 'roi_heads'):
            logger.info('eval box head')
            model.module.roi_heads.box.eval()
        if fix_mask_head and hasattr(model.module, 'roi_heads'):
            logger.info('eval mask head')
            model.module.roi_heads.mask.eval()
        if fix_shape_head and hasattr(model.module, 'roi_heads'):
            logger.info('eval shape head')
            model.module.roi_heads.shape.eval()
        if fix_disparity and hasattr(model.module, 'dispnet'):
            logger.info('eval dispnet')
            model.module.dispnet.eval()
        if fix_pointrcnn_rpn and hasattr(model.module, 'pcnet'):
            logger.info('eval pointrcnn rpn')
            model.module.pcnet.rpn.eval()
        if fix_pointcloud and hasattr(model.module, 'pcnet'):
            logger.info('eval pcnet')
            model.module.pcnet.eval()
    return model


def retreive_params(cfg):
    fix_backbone = cfg.SOLVER.FIX_BACKBONE
    fix_rpn = cfg.SOLVER.FIX_RPN
    fix_box_head = cfg.SOLVER.FIX_BOX_HEAD
    fix_mask_head = cfg.SOLVER.FIX_MASK_HEAD
    fix_shape_head = cfg.SOLVER.FIX_SHAPE_HEAD
    fix_disparity = cfg.SOLVER.FIX_DISPARITY
    fix_pointrcnn_rpn = cfg.SOLVER.FIX_POINTRCNN_RPN
    fix_pointcloud = cfg.SOLVER.FIX_POINTCLOUD
    if all([fix_backbone, fix_rpn, fix_mask_head, fix_box_head, fix_shape_head, fix_disparity, fix_pointrcnn_rpn,
            fix_pointcloud]):
        raise ValueError('must open one.')
    return fix_backbone, fix_rpn, fix_box_head, fix_mask_head, fix_shape_head, fix_disparity, fix_pointrcnn_rpn, fix_pointcloud
