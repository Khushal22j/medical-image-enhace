# medical-image-super-resolution
 
In this project, we use the Real-Enhanced Super-Resolution Generative Adversarial Network (Real-ESRGAN) model for medical image super-resolution.
In our proposed approach, the pre-trained generator and discriminator networks of the 
Real-ESRGAN model are fine-tuned using medical image datasets.

In this project, we worked on retinal images - chest X-ray images. We used the STARE dataset of retinal images and Tuberculosis Chest X-rays (Shenzhen) dataset.
Our fine-tuned model produces more accurate and natural textures, and the output images have better detail and resolution compared to the original real-esrgan model.

datasets can be downloaded from the link below: 

STARE dataset :

https://www.kaggle.com/datasets/vidheeshnacode/stare-dataset

Tuberculosis Chest X-rays (Shenzhen) dataset:

https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-shenzhen

All details and fine-tuning parameters are available in the finetune_realesrgan_x4plus.yml file.

It is also possible to test the original real-esrgan model and fine-tuned model in the fine_tune_retinal_image.ipynb and fine_tune_chest_x_rays.ipynb files

Below are some outputs of the real-esrgan model and the fine-tuned model:

Retinal images

![1](https://user-images.githubusercontent.com/47056654/197783863-8a03ac44-163a-4675-965d-f1d751e2f4a2.jpeg)

![2](https://user-images.githubusercontent.com/47056654/197783906-ae4c5ae2-85bd-48bb-a9ea-9d7826c60625.jpeg)

![3](https://user-images.githubusercontent.com/47056654/197783926-2bc39b44-e047-4e13-92e1-3bd4c2891141.jpeg)

![4](https://user-images.githubusercontent.com/47056654/197783954-dfe2c0b1-9cba-4359-85e2-7d01af913e7d.jpeg)


Chest X-ray images 

![5](https://user-images.githubusercontent.com/47056654/197784017-a40d6ddd-baca-4a0c-8406-c33c65c527b2.jpeg)

![6](https://user-images.githubusercontent.com/47056654/197784062-8025888f-7873-4017-8ee6-797dbf7d9de7.jpeg)

![7](https://user-images.githubusercontent.com/47056654/197784093-19a0c6cd-335a-4f7a-b871-c30bcf6f8042.jpeg)








