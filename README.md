# X_to_Y_Prediction
A cute project to predict a Y value based on an X value. Works best with Linear relationships. This is by far the easiest project you can try out in order to get into machine learning as it doesn't make use of any fancy optimizing functions- and only uses one weight. Not even a bias is used. Everything is made easy to understand just by reading the code except for the backward function here: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html

The backward function takes the gradient (Derivative or slope in Calculus) of the loss and is then used to put on the graph (red line). As the loss function decreases we get closer to a meaningful prediction. At the same time the weight change (blue line) __converges__ towards where we want it. 

Need Pytorch, matplotlib in order for this to work. GPU isn't needed nor used. Use the CPU and just do it!
