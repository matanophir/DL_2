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
A. No it wouldn't produce a gradient equivalent to GD because in the forward pass we only calculate the needed function and dont accumulate anything, so in practice the gradients will be lost.\

"""

part2_q4 = r"""
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
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
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