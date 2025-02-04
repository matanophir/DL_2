r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**\
 1.\
 A. we can look at $\pderiv{\mat{Y}}{\mat{X}}$ as a matrix D with the shape of X- (N,$D_{in}$) where every element $d_{ii} $ is $\pderiv{\mat{Y}}{\mat{X_{ii}}}$. this also yields a matrix $D2$ with the size of Y - (N, $D_{out}$).\
 overall we get $(N, D_{in}, N, D_{out}) = (64,1024,64,512)$\
 \
B. Yes because $\pderiv{\mat{Y}_n}{\mat{X}_m}$ is zero whenever n /= m.\

C. Thanks to the sparsity mentioned above, among other things, there's no really need in the whole Jacobian which would be ofc infeasible. Rather,\
We can obtain $\pderiv{\mat{Y}}{\mat{W}} = \pderiv{\mat{Y}}{\mat{X}} * \pderiv{\mat{X}}{\mat{W}}$ by equivalently using:\
$Y = XW^T + b$ -> $\delta\mat{X}=\delta\mat{Y} * \mat{W}$\
\
2.\
A. following the logic above we can conclude that $\pderiv{\mat{Y}}{\mat{W}}$ is of size $(512,1024,64,512)$.\
\
B.
$\mat{Y}_{n,j}$ is affected only by $\mat{W}_{j,i}$ (for all i). Thus, $\pderiv{\mat{Y}}{\mat{W}}$ is sparse as every element $(n,k,j,i)$ which represents $\pderiv{\mat{Y}_{n,k}}{\mat{W}_{j,i}}$ is zero whenever $k!=j$.\
\
C. No. Again, similarly: $\delta{W} = \delta{Y}*X$\
 
"""

part1_q2 = r"""
**Your answer:**
Generally preferred but this is not a must. We've talked in class about some alternatives, in particular - even computing $\delta\mat{X}$ by hand is possible. This quickly will get messy and ofc isn't scalable but possible...\
"""

# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.01
    reg = 0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr_vanilla = 0.02
    lr_momentum = 0.002
    lr_rmsprop = 0.00015
    reg = 0.01
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.002
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**\
1.Yes it matches what we expected. We expected poorer results on the training set but better results on the test set.\
Introducing dropout makes the model to generalize better at the cost of added noise in the training phase.\
\
2. As expected higher drop-out leads to poorer performance on the training set because of the added noise (the model keeps changing and it makes it difficult learning).\
It also seems that highly regularized model (high dropout) struggles to generalize and preform better than the 0.4 model on the test set.\
Interesting to see that the 0.8 model has lower avg loss on the test set but also lower test_acc. We're using CE loss and this finding may indicate that maybe the 0.8 was overall closer to the true labels but was less confident- reminds of high regularization that "smoothens" (underfitting) the decisions.

"""

part2_q2 = r"""
**Your answer:**\
Yes it can happen and we can see an example for this by looking at the 0.4 model from iteration 10.\
it can happen when the model becomes less confident (i.e gives lower probability to answers) overall (and so also on the right answer) lowering the $log(softmax(x))$ term, but manages to still give the highest probability to the right answers.

"""

part2_q3 = r"""
**Your answer:**\
1. GD and back-propagation are two things from different but connected domains.\
GD is a method to optimize a model by using the gradients w.r.t the parameters, and one way of obtaining them is using back-propagation (and the chain rule) to calculate them with ease.\
\
2. As stated earlier: let $N$ be the total training samples and $M$ the size of batch we are working on-\
- If $M=N$ this is known as regular gradient descent. If the dataset does not fit in memory the gradient of this loss becomes infeasible to compute.\
- If $M=1$, the loss is computed w.r.t. a single different sample each time. This is known as stochastic gradient descent.\
- If $1<M<N$ this is known as stochastic mini-batch gradient descent. This is the most commonly-used option.\
So it comes down to the number of samples we're working on between steps of the optimizer.\
Stochastic versions may be better for optimization because we get a dynamic loss surface which help the optimizer to don't get stuck in local minimums.\
\
3. There can be few reasons why stochastic version are more common. First and foremost many times the dataset can't fit entirely into memory so it is required to split in order to work with, leading to stochastic version.\
Secondly, as stated above the stochastic version may lead to better optimization by avoiding local minimums.\
we also think that the stochastic version makes it harder for the model to over-fit and thus might be better foo generalization.\
\
4.\
A. No it wouldn't produce a gradient equivalent to GD because in our implementation the forward pass we only calculate the needed function and doesn't accumulate anything, so in practice the gradients (or the needed inputs at every step) will be lost.\
If we were to change the implementation so that we will save the needed state at every step or maybe avg it, it may work.
If the question actually ask to do forward and backward for every batch (without zero_grad) and only make one final step with the optimizer- it would work because at each back-propagation we sum the gradients w.r.t each parameter and from the linearity of summation it's equal.\
\
B. Depends on the implementation. if we chose to save the state at each forward pass in order to later calculate the gradient, we will have to save a lot of data so we would run out of memory. 
"""

part2_q4 = r"""
**Your answer:**\
1.\
A. We will notice that for the forward mode we only need the last computed gradient ($\pderiv{v_j}{v_0}$) and $v_j.val$ to multiply with $\pderiv{v_{j+1}}{v_j}$ so we don't actually need to store values on the nodes (other than the last one) so it's $O(1)$ memory.\
B. If we would calculate the gradients with the chain rule we would need to store all the intermediate values so the memory complexity would be $O(n)$.\
    We can avoid this if we have some assumptions such as all the functions are easily invertible and then we can perform backward pass and have both the value and the grad in each "step" acheiving O(1) memory complexity (only ~current "step" matters) and O(n) computation.\

We can relax the assumption with the following idea:\
We'll have a fusion of the "improved" backward AD (where possible - on invertible functions) and  "improved" forward pass (without saving intermediate values by default but in specific nodes as explained later).\
Firstly, we do a single "improved" forward pass and when we encounter univertible function (or "hardly invertiable") we store the intermediate value, and proceed that way.\
 By the end of the pass we have the last value, and we can implement the improved backward AD as long as we're not encountering uninvertible function - when we do, we use the corresponding stored intermediate value and continue the backward AD as usual.\
 This way we reduce memory where we can and use the "ordinary" usage of memory (storing intermediate value in the node) where not.\

2.\
 the forward AD technique can be generalized though be useful mainly when we have only one starting node because otherwise we would need to compute the forward AD w.r.t every starting node which would make it inefficient.\
We can generalize the 'improved' backward AD with the usual caveats of topology and its affect on computation. graphs with many uninvertible or complex functions nodes will hinder the ability to save memory resulting in unwanted added complexity.\

3. our implementation saves intermediate values where it can't compute them with the inverse function (doesn't exist or too complex), so in practice saves memory compared to the regular backward AD.\
in deep networks we have many computational steps thus we have more opportunities to save memory with our improvement.\
    



"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**\
1. optimization error is the error that occurs due to the training process. e.g if we are training with gradient decent and the loss is non-convex we might get stuck on a local minima and not converge to the global minimum -> optimization error.\
we can decrease this error by using different maybe more suitable optimizers (different hyperparameters, algorithm..)
\
2. generalization error occurs because We train to minimize our empirical loss instead of the population loss. To reduce this error we can various method to try and prevent the overfitting such as smaller model, data augmentation, regularization (L2 or dropout), using validation set etc.\
\
3.  Approximation error - indicates the model's inability to represent the target function due to its insufficient complexity. We can lower it by increasing model complexity through width and/or depth, as well as changing architecture - adjust receptive field (the idea - better features or changing hypothesis set (model), etc.\
"""

part3_q2 = r"""
**Your answer:**
Say we are trying to classify ones from a set of numbers between 0-9.
We would have a high false positive rate if in this dataset the sevens are very similar to a one. we would classify all the sevens as ones.\
We would have a high false negative rate if all the ones are not drawn similarly enough to one another and the classifier can't differentiate them from the other numbers.
"""

part3_q3 = r"""
**Your answer:**
We have no ROC curve but we assume the optimal point is the one that balances FPR and FNR.\
1. in this scenerio we have no cost to have a false negative situation because the patient will develop non-lethal symptoms that will show he has the disease. so we only diagnose a patient with disease if we have high certainty.\
2. now we high cost for false negative so we would diagnose disease even with low certainty, thus risking in high FPR but at the cost a saving lives. 
"""


part3_q4 = r"""
**Your answer:**
In MLP every data point passes through the network and the loss is calculated w.r.t that point. in a sentence words has connections (i.e order) them that must be evaluated but unfortunately lost due to the way MLP works.\
So an MLP will predict the sentiment based on each word separately.
"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    lr = 0.01
    weight_decay = 0.0001
    momentum = 0.9
    loss_fn = torch.nn.CrossEntropyLoss()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**
1. the formula for the number of paramaters in a conv layer is: $K * (C_{in}*F^2 + 1)$ 
so for the left block (when we replace 64 with 256):
- first layer params = $256 * (256*3^2 +1) = 590080$
- second layer is the same = 590080
- total = 1180160

for the right block:
- first layer = $64 *(256*1^2 +1) = 16448$
- second layer = $64 * (64*9 + 1) =  36928$
- third = $256 * (64*1 +1) = 16640$
- total = 70016

2. the amount of operations in a layer from ${C_{in},H_{in},W_{in}}$ to $C_{out},H_{out},W_{out}$ is $2 * C_{in} * F^2 * C_{out} H_{out} * W_{out}$
where F is the size of the kernel and the '2' is for sum and mul.\
in our implementation we kept the spatial dim so we can see the number of operations are mainly dependant on the number of channels which the left block has much more of.

3. the regular block uses 3x3 directly on the 256 feature map so its able to combine the input both spatially and across feature maps ($(3,3,C_{in})$ kernels).\
the bottle-neck block first reduces dimensionality and then does the 3x3 conv on the reduced form which means its ability to combine input spatially is a bit worse.\
regarding the ability to combine across feature maps, we think it still retains high ability and this is because of the 1x1 convs that combines the input across feature maps.


"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**
1.it seems depth can be a real problem to CNNs as it can make a model untrainable. the 'shallower' models (L2 L4) were able to learn while the deeper ones didn't manage.\
between the models who learned the difference seems negligible but the model who scored the best was L4_64K. we assume thats because he had more expressive power (more kernels) without the risk of over-fitting (highish reg 1e-2) while still able to control the gradient flow.\
\
2. yes there were values of L for which the network wasn't trainable- L8, L16. we think it's mainly because of vanishing gradients. this may be resolved with batch normalization to stabilize gradients along the way and ofc adding residual connections.

"""

part5_q2 = r"""
**Your answer:**
So we've seen in 1.1 that the net can handle about 4 layers and more kernels made it better presumably because of the added expressiveness (with sufficient regularization).\
here our ideas get reaffirmed as the model is unteachable at L8 and the best result came from the model with the most kernels (and possible layers) L4_K128.\

"""

part5_q3 = r"""
**Your answer:**
yet again we see that the model can handle up to 4 layers as it failed at L3_k64-128 (6 layers) presumable because of the gradient flow or lack thereof.\
worth noting the L2_k64-128 did worse that L4_128. the added expressiveness works in our favour here.

"""

part5_q4 = r"""
**Your answer:**
now we see that the net is still trainable with a lot more layers and actually the model with the most layers L32_K32 out-preformed (score and stability but very close to L2/L4_K64_-128-256) other models.\
we can see that as we increased the kernel sizes the model scored almost the same but became less stable. noticing that when the number of kernels get very large (L8_k64-128-256) We are assuming that the poorer results also relates to the gradients but in a different way- the gradient is distributed across all these parameters hence each one get smaller and maybe leads to very slow learning rate that coupled with the early stopping mechanism might lead to problems.
quick calculation show that L4_k64-128-256 has about $4*64*64*9 + 4*128*128*9 + 4*256*256*9 = 1327104$ parameters and L32_K32 has about $32*32*(32*9) = 294912$ which might explain it but we need to test some more stuff like adding batch normalization, residual paths more frequently (not just after pooling which happens every 8th block), tweaking the learning rate and increasing early stopping param to see how it behaves.

"""

part5_q5 = r"""
"""

# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**
1. The model did a pretty bad job at detecting the objects.\
the bounding boxes did manage to bound objects but the class prediction was very poor and the confidence is low accordingly.\

2. in the first image of dolphins one possible cause is training dataset bias:\
- the model was mostly shown during the training process surfer in the ocean and therefore without distinct features of something else, it will classify an object in that scenery as a surfer.\
- the image is missing its texture and maybe it something the model didn't see before.\
in the dog image there could be a problem with how the model assigns bounding box and the following classification.\
if the model doesn't intersect the image good enough and left to process big bounding boxes that may contain more than on class, then it might confuse the model as we see here.\

3. yolo is trained end to end on a single neural network so we can do what we have seen in tutorial 4 PGD attack.\
    by taking an image and maximize its loss where the input parameter (which we are optimizing) is a delta from the original image.

"""


part6_q2 = r"""
**Your answer:**


"""


part6_q3 = r"""
**Your answer:**
actually the model did pretty good when given reasonable tasks (see cow on the beach or moving man).\
when given more difficult tasks such as an almost completely hidden banana or a blurred plant it didn't succeed in classifying then.
We can assume the following pitfalls caused the mistakes:\
model bias + deformation - ocean-cow\
occlusion - hidden_banana\
blurring - blurred_plant.

"""

part6_bonus = r"""
**Your answer:**
Tried fixing the other photos but failed. we've resharpen the image,
"""