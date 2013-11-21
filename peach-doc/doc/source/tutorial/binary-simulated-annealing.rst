Optimization by Binary Simulated Annealing
==========================================

The simulated annealing is actually a meta-heuristic to drive the optimization.
There is nothing to say that it can't be used to optimize functions of discrete
variables. In fact, simulated annealing excels in this kind of optimization.
Peach provides a discrete optimizer based on arrays of bits. Notice that you
must have the ``bitarray`` module installed so you can use this.

The discrete simulated annealing behaves basically in the same way that the
continuous simulated annealing algorithm. The instantiation, however, is a bit
different. We will get there soon. As before, we will use the algorithm to find
the minimum of the Rosenbrock function. We start by importing ``peach`` and
``numpy`` in different namespaces::

    from numpy import *
    import peach as p

We must also create the objective function for the algorithm::

    def f(xy):
        x, y = xy
        return (1.-x)**2. + (y-x*x)**2.

Notice that, even though the optimizer behaves in a different way, you define
your function in pretty much the same way, if you want to. But the discrete
optimizer is more powerful, as you can, if you so wish, receive not floating
point numbers in your objective function, but integers, characters or even the
bitarray itself, so you can decode it anyway you want to. More on this soon.

Now we will create the optimizer. We create these optimizers in the same way we
created other optimizers: by instantiating the corresponding class, passing the
function and the first estimate. Notice that the first estimates are given in
the form of a tuple, with the first estimate of :math:`x` in the first place,
and the first estimate of :math:`y` in the second place. There is no need to use
tuples: lists or arrays will do. To create the optimizers, we issue::

    dsa = p.DiscreteSA(f, (0.1, 0.2), 'ff', [ (0., 2.) ])

The creation of the optimizer is somewhat different than in previous occasions.
The first two parameters are the ones we are used to see: the objective function
to be optimized, and the first estimate of the location of the minimum. The next
two need some explanation.

In the discrete optimizer, the estimate is not encoded as an array, but as an
array of bits. Unfortunatelly, there is no unique way to interpret what the bits
mean, so we must inform the algorithm how to deal with them. The third argument
to the instantiation of the class is exactly what does this: it is a string of
characters that inform what kind of value will be in which place. Although it
seems quite cryptical, it is very simple: use 'L' for each integer, and 'f' for
each float. The object will decode the bit stream and send to your objective
function a tuple containing the decoded values. In fact, you could use it
differently: the decoding is based on the ``struct`` module, which is standard
in every Python installation. Please, consult the official documentation for
more information. Here, we will be using two floating points, so the format is
``'ff'``.

The next parameter lists the ranges of allowed values for each variable. This
should be informed if you are using floating points, since there are sequences
of bits that do not decode to valid numbers. The algorithm will use this
information to check if the decoded values are legal and, if not, to random
choose new estimates. While this randomness might seem strange, it is a good
thing -- remember, the simulated annealing is an stochastic optimization method.
If you, however, are dealing only with integers or you don't want the algorithm
to make this check, you don't need to inform ranges at all.

There are some ways to inform the algorithm what are the ranges. A range, in
this context, is a two-tuple of values, ``(x0, x1)``, where ``x0`` is the lower
limit and ``x1`` is the higher limit of the interval. The ranges must be given
as a list, each range applying to the respective variable (that is, the first
range applies to the first variable in the optimization, the second range to the
second variable and so on). If you, however, use only one range in the range
list (as in this case), the same range will apply to every variable. In this
case, all variables will range from 0 to 2, thus ``[ (0., 2.) ]``.

As we done in the other optimization tutorials, we will execute the algorithm
step by step. We can do this to keep track of the estimates to plot a graphic.
We do this using the commands::

    xs = [ ]
    i = 0
    while i < 500:
        x, e = dsa.step()
        xs.append(x)
        i = i + 1

Notice that we used 500 iterations here. In general, stochastic methods pay this
price to be able to find the global minimum: they need more iterations to
converge. That's not a problem, however, since finding the global minimum is
a desired result, and the penalty in the convergence time is not that
significant. But discrete simulated annealing executes so incredibly fast that
this won't be a problem in any way.

The ``xs`` variable will hold, in sequence, the estimates. We can plot them to
see the convergence trace. The figure below is a representation of the execution
of the method. The function itself is represented as contour curves in the
plane, and the estimate tracks over them. Notice the unusual path that the
estimate followed to arrive at the result.

.. image:: figs/discrete-simulated-annealing.png
   :align: center
