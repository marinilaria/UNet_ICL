# UNet_ICL

This package allows to work on the fits files to modify the images in different ways. It should be general enough to work with fits files coming from different output, as long as the data is contained in the first index (see init function in Smac_Map).

It also contains the UNet to use for training and testing magnitude.
## Minimum working code:


```
#!/usr/bin/env python
# coding: utf-8


import UNet_ICL.fits as fits

#create an instance of the transformation
ICL_transformed = fits.Transformation_Map("D2_BCG+ICL.0.0.066.0.x.ICL.HST_ACS-WFC_F814W.fits")
BCG_transformed = fits.Transformation_Map("D2_BCG+ICL.0.0.066.0.x.BCG.HST_ACS-WFC_F814W.fits")


#Resize image
random_crop = fits.random_size(ICL_transformed.get_size(), pixels=512)
ICL_resized = ICL_transformed.resize_image(crop = random_crop)
BCG_resized = BCG_transformed.resize_image(crop = random_crop)

#Draw random ellipse
mask = fits.get_mask_ellipse(ICL_resized, N=1, n=5)
ICL_perturbed = ICL_resized.draw_ellipse(mask)
BCG_perturbed = BCG_resized.draw_ellipse(mask)

#Rotate
BCG_perturbed = BCG_resized.apply_perturbations("rotate", 1)
ICL_perturbed = ICL_resized.apply_perturbations("rotate", 1)

#create magnitude map
ICL_magnitude = ICL_perturbed.to_magnitude()
BCG_magnitude = BCG_perturbed.to_magnitude()

#create numpy object
ICL_magnitude = ICL_magnitude.get_data()
BCG_magnitude = BCG_magnitude.get_data()

plt.imshow(ICL_magnitude)
plt.show()

plt.imshow(BCG_magnitude)
plt.show()

```
