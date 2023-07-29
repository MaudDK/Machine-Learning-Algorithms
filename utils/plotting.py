import matplotlib.pyplot as plt
import numpy as np

def rnd_sample_plot(X, y):
    m, n = X.shape
    fig, axes =- plt.subplots(8,8, figsize=(5,5))
    fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]

    for i, ax in enumerate(axes.flat):
        random_index = np.random.randint(m)
        #Reshape Image to 20x20 pixels
        X_random_reshaped = X[random_index].reshape((20,20)).T
        ax.imshow(X_random_reshaped, cmap='gray')
        ax.set_title(y[random_index,0])
        ax.set_axis_off()
    
    fig.suptitle("Label, image", fontsize=14)