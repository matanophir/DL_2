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
 B. Yes most of the elements are zero because every output $Y_i$ is reliant only on the respective input $X_i$ so at every $D2$ only one row will be non-zero.\
 \
 C. We don't need to materialize the above Jacobian to calculate $\pderiv{\mat{Y}}{\mat{X}}$ because we only the need the product $\pderiv{\mat{L}}{\mat{Y}} * \pderiv{\mat{Y}}{\mat{X}}$ thanks to the chain rule.\
since $\pderiv{\mat{Y}}{\mat{X}} = \mat{W}$ we can just compute $\pderiv{\mat{L}}{\mat{X}} * \mat{W}$.\
\
2.\
A following the logic above we can conclude that $\pderiv{\mat{Y}}{\mat{W}}$ is of size $(512,1024,64,512)$.\
\
B. Same here. the Jacobian is sparse because only one score in a given y is reliant on some $w_{ii}$, so in every $D2$ matrix we will have only one non-zero column.\
in simpler terms every col in $\mat{Y}$ depends on one row in $\mat{W}$.\
\
C. Again we dont need. we can just compute $\pderiv{\mat{Y}}{\mat{W}} = \mat{X}$ and use the chain rule to compute $\delta\mat{W}$
 
"""

part1_q2 = r"""
**Your answer:**
 back-propagation is not required because we can just compute $\delta\mat{X}$ by hand.\
 this straight on approach will get very messy very quickly and it wouldn't be modular, i.e we would have to calculate again with every change we introduce to the architecture.
 


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
B.? We can put 'checkpoints' of calculated gradient (forward AD) so we wouldn't need to save all the activations in the forward pass. and then when calculating the backward pass we will use those checkpoints.\
maybe truncate the nodes into one by calculating the analytical derivative?

2. It can be under certain constraints. the forward pass technique can be useful mainly when we have only one parameter because otherwise we would need to compute the forward pass w.r.t every parameter which would make it inefficient.\
    ?
3. ?



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
1. optimization error is the error that occurs due to the training process. e.g if we are training with gradient decent and the loss is non-convex we might get stuck on a local minima and not converge to the global minimum $\if$ optimization error.\
we can decrease this error by using different maybe more suitable optimizers such as rmsprop or momentum.\
\
2. generalization error occurs because We train to minimize our empirical loss instead of the population loss. To reduce this error we can various method to try and prevent the overfitting such as regularization and dropout.\
\
3. approximation error occurs when we only consider a limited set of possible function. due to the approximation law we rarely have high approximation error because was shown that MLP with one hidden layer can approximate every function.
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

2. the amount of operations in a layer from ${C_{in},H_{in},W_{in}}$ to $C_{out},H_{out},W_{out}$ is $2 * C_{in} * F^2 * C_{out} H_{out} * W{out}$
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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""