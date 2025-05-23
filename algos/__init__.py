from .FaithDaIL import FaithDaIL



def ALGO(args, env=None):
    if args.algo == 'FaithDaIL':
        print("Using FaithDaIL!!")
        return FaithDaIL(args, env)
    else:
        raise ValueError(f"Invalid algorithm: {args.algo}")