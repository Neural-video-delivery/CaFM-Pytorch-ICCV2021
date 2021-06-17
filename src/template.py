def set_template(args):
    # Set the templates here
    if args.template.find('jpeg') >= 0:
        args.data_train = 'DIV2K_jpeg'
        args.data_test = 'DIV2K_jpeg'
        args.epochs = 200
        args.decay = '100'


    if args.template.find('RCAN') >= 0:
        args.model = 'RCAN'
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 64
        args.chop = True

    if args.template.find('EDSR') >= 0:
        args.model = 'EDSR'
        args.n_resblocks = 16
        args.n_feats = 64

    if args.template.find('VDSRR') >= 0:
        args.model = 'VDSRR'
        args.n_resblocks = 20
        args.n_feats = 64
        args.patch_size = 48
        args.lr = 1e-4

    if args.template.find('ESPCN') >= 0:
        args.model = 'ESPCN'
        args.n_feats = 64
    
    if args.template.find('SRCNN') >= 0:
        args.model = 'SRCNN'
        args.n_feats = 64