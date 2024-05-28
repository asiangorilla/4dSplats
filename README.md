# 4dSplats
Remote Sensing Project TU Berlin Summer Semester 2024

relevant papers

3D Gaussian Splats (main paper)
https://dynamic3dgaussians.github.io/

4D Gaussian
https://guanjunwu.github.io/4dgs/

Dynamic-NeRF (deformation function)
https://www.albertpumarola.com/research/D-NeRF/index.html

Gaussian Splats:
https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/



Concept of our Model:
  We want to use gaussian splatting in a similar manner as the dynamic gaussian splats paper (3D Gaussian Splats). But except for having Pixel X Time many gaussians, we want to have pixel many gaussians and save the change of the gaussian with time as a ML function F(t, g) -> g'. 

Idea of Model:
  We combine Gaussian Splats together with dynamic nerf, by using the gaussian splat model with pretrained weights for the initial frame (similar to dynamic gaussians). Using the result from the first frame gaussian splats, we keep the standard deviation, color, opacity logit and the background logit constant. We assume temporal change only for the gaussian centers (x,y,z) and the quaternions (w,x,y,z). For the temporal change, we use another MLP, analogous to the Dynamic-Nerf paper.

Deformation Model Architecture:
  Dynamic-Nerf uses the cartesian coordinates for the pixels as an input to the deformation netowrk. While the paper is not completly transparent about the architecture of the deformation model, one can assume, that the build is similar to the Nerf network. SInce we have two different inputs (cartesian coordinates and quaternions), we have multiple options for our model architecture. Two main ideas are listed below. The main Model architecture is shown in Diagram 1.

![Diagram1](https://github.com/asiangorilla/4dSplats/assets/67825736/95e59279-5206-40a4-9891-e33978fe51c3)

The base model shown in Diagram 1 shows the calculation of the delta-cartesian, with RELU activation between the linear layers. The final output does not have any non-linear function. The first option is to sepearatly calculate both the delta-cartesian and the delta-quaternion. FOr this, we essentialy have two models with the same architecutre, the only difference being the output and input dimensions. Another possibility we have tried is to combine both delta calculations in one model. FOr this, we have one model with an input dimension of 30 + 60 = 90 (60 being the expanded quaternion deimension) and output dimension being 3+4 = 7. The model additionally ensures that the outgoing quaternion is a unit quaternion, so that the resulting quaternion can be immediatlly used to calculate the resulting quaternion q(t).

Loss function: 


Gaussian Splat Model -> deformation network:


deformation Network output -> visual output:
