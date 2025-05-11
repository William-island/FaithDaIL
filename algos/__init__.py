from .online_drm_odice_isw import ONLINE_DRM_ODICE_ISW



def ALGO(args, env=None):
    if args.algo == 'online_drm_odice_isw':
        print("Using online_drm_odice_isw_wob!!")
        return ONLINE_DRM_ODICE_ISW(args, env)
    else:
        raise ValueError(f"Invalid algorithm: {args.algo}")