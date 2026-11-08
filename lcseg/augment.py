"""Online augmentation of the training images (Section III.G, Table 4).

Horizontal and vertical flips, rotations by multiples of 90 degrees, small shifts,
scalings and rotations, elastic deformation, brightness and contrast changes,
hue-saturation-value shifts, CLAHE, and Gaussian noise. Geometric transforms are
applied to the mask as well (albumentations does this automatically for `mask`).
The manuscript does not list probabilities or magnitudes; the values below are
the ones used by this code and are recorded in every run.json.
"""

AUG_PARAMS = {
    "hflip_p": 0.5, "vflip_p": 0.5, "rot90_p": 0.5,
    "ssr": {"shift": 0.05, "scale": 0.05, "rotate": 15, "p": 0.3},
    "elastic_p": 0.2,
    "brightness_contrast": {"limit": 0.15, "p": 0.3},
    "hsv": {"hue": 8, "sat": 15, "val": 10, "p": 0.3},
    "clahe_p": 0.1,
    "noise_p": 0.1,
}


def build_augmenter(params=AUG_PARAMS):
    import albumentations as A
    import cv2
    ssr = params["ssr"]
    try:   # albumentations >= 1.4 provides Affine with the same effect as ShiftScaleRotate
        geo = A.Affine(translate_percent=(-ssr["shift"], ssr["shift"]),
                       scale=(1 - ssr["scale"], 1 + ssr["scale"]),
                       rotate=(-ssr["rotate"], ssr["rotate"]),
                       border_mode=cv2.BORDER_REFLECT_101, p=ssr["p"])
    except TypeError:
        geo = A.ShiftScaleRotate(shift_limit=ssr["shift"], scale_limit=ssr["scale"],
                                 rotate_limit=ssr["rotate"], border_mode=cv2.BORDER_REFLECT_101,
                                 p=ssr["p"])
    bc, hsv = params["brightness_contrast"], params["hsv"]
    return A.Compose([
        A.HorizontalFlip(p=params["hflip_p"]),
        A.VerticalFlip(p=params["vflip_p"]),
        A.RandomRotate90(p=params["rot90_p"]),
        geo,
        A.ElasticTransform(p=params["elastic_p"]),
        A.RandomBrightnessContrast(brightness_limit=bc["limit"], contrast_limit=bc["limit"], p=bc["p"]),
        A.HueSaturationValue(hue_shift_limit=hsv["hue"], sat_shift_limit=hsv["sat"],
                             val_shift_limit=hsv["val"], p=hsv["p"]),
        A.CLAHE(p=params["clahe_p"]),
        A.GaussNoise(p=params["noise_p"]),
    ])
