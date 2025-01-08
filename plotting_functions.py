import numpy as np
from matplotlib import pyplot as plt

def plot_decision_boundary(model, ax, X, title, alpha):
    
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    
    x = np.linspace(x_min,x_max, 100)
    y = np.linspace(y_min, y_max, 100)

    xv, yv = np.meshgrid(x,y)
    
    points = np.column_stack((xv.ravel(), yv.ravel()))
    
    labels = model.predict(points).reshape((100,100))
    #plt.contourf(x,y,labels,cmap='RdBu')
    #plt.show()
    ax.contourf(x, y, labels, cmap='viridis', alpha=alpha)
    ax.set_title(title)
    ax.grid(True)

def plot_mean_std(ax, data, x, label, color, linestyle='-'):

	mean = np.mean(data, axis=0)
	std = np.std(data, axis=0)

	lower_bound = np.maximum(mean - std, 0)  # Prevent negative values
	upper_bound = mean + std

	ax.plot(x, mean, label=label, color=color, linewidth=2, linestyle=linestyle)
	ax.fill_between(x, lower_bound, upper_bound, color=color, alpha=0.3, label=f"±1 Std Dev {label}")