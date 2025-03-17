from .model import RAM_Model



def build_model(cfg, clip_model, dataset, clip_model_teacher=None):
    model = RAM_Model(
        cfg, 
        clip_model, 
        dataset.classnames_seen,
        dataset.classnames_unseen,
        clip_model_teacher
    )
    print('Build RAM Model Done')
    return model
    