import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def create_collage():
    results_dir = 'results'
    images = [
        'cm_Logistic_Regression.png',
        'cm_Naive_Bayes.png',
        'cm_Linear_SVC.png',
        'cm_Random_Forest.png',
        'cm_Decision_Tree.png',
        'cm_Voting_Ensemble.png'
    ]
    
    # Filter only existing images
    images = [img for img in images if os.path.exists(os.path.join(results_dir, img))]
    
    if not images:
        print("No confusion matrix images found in results/")
        return

    num_images = len(images)
    cols = 2
    rows = (num_images + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    
    for i, img_name in enumerate(images):
        img_path = os.path.join(results_dir, img_name)
        img = mpimg.imread(img_path)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(img_name.replace('cm_', '').replace('.png', '').replace('_', ' '), fontsize=16)
        
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrices_collage.png'), dpi=150)
    print(f"Collage saved to results/confusion_matrices_collage.png")

if __name__ == "__main__":
    create_collage()
