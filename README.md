# Linear Shallow Water Model

- **Typical use:** Geostrophic adjustment.
- **Original code by:** James Penn, modified by G. K. Vallis
- **Object-oriented version by:** G. Giuliani
- **Description:** Two-dimensional shallow water model in a rotating frame.
- **Grid Type:** Staggered Arakawa-C grid.
- **Boundary Conditions:** Fixed boundary conditions in the y-dimension (free slip).
- **Linearization:** About a fluid depth H and u = 0.

**Dimensions (SI units):** Implied via values of constants. For instance, Lx is the width of the domain in meters (m), but the code is not specifically dependent on any units. If all input values are appropriately scaled, other units may be used.

### Equations:
$$
\begin{align*}
\frac{\partial u}{\partial t} - fv &= - g \frac{\partial h}{\partial x} + F \quad \text{(1)} \\
\frac{\partial v}{\partial t} + fu &= - g \frac{\partial h}{\partial y} + F \quad \text{(2)} \\
\frac{\partial h}{\partial t} + H\left(\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}\right) &= F  \ \ \ \ \ \ \ \ \ \ \ \ \ \qquad \text{(3)} \\
\eta &= H + h \ \ \ \ \  \qquad \text{(4)}
\end{align*}
$$
- Here, F is a forcing term, with the default value F = (0, 0, 0).

## Usage:

The code provides flexibility for multiple cases. Below is a template to initialize and run the model for different scenarios:

```python
# Example Usage
%run ShallowWaterModel.ipynb # If you are using the Jupyter Notebook version
import ShallowWaterModel # If you are using the .py version (Make sure the module is in the same directory of your script)

# Initialize the model with the desired case
case = "Your Desired Case"
model = ShallowWaterModel(case)

# Run the model
model.model_run()

# Access results
u, v, h, t = model.download_arrays()

# Plot the results at any given instant of time T
model.plot_all(u[:,:,T], v[:,:,T], h[:,:,T], t[T])

# Plot an animation of the ongoing simulation (pre-defined step = 2, but can be changed)
model.anim_simulation() # for a custom step, model.anim_simulation(step)

# Plot an Hovmöller diagram of the simulation
model.Hovmuller()
​
```

A list of the possible already implemented cases is provided: 
- Non-Rotational Waves (``` "Non-Rotational Waves" ```)
- Rotational Waves on f-plane at 20°N (``` "Rotational Waves (f)" ```)
- Rotational Waves on $\beta$-plane at 20°N (``` "Rotational Waves (beta)" ```)
- Equatorially trapped waves - small disturbance (``` "Equatorially-trapped Waves (s)" ```)
- Equatorially trapped waves - large disturbance (``` "Equatorially-trapped Waves (l)" ```)
- Equatorially trapped waves - all walls (``` "Equatorially-trapped Waves (a)" ```)
​

​
However, it is possible to create your custom-made case (``` "Custom-Made case" ```). For this, you can modify any parameter and those that remain unchanged will 
take the corresponding values from the default simulation, namely the (``` "Non-Rotational Waves" ```). Here, I provide a brief snippet for this case:

```python
# Example Usage
%run ShallowWaterModel.ipynb # If you are using the Jupyter Notebook version
import ShallowWaterModel # If you are using the .py version (Make sure the module is in the same directory of your script)
​
# I want to simulate a shallow-water environment in a rectangular domain with reflecting boundaries
model = ShallowWaterModel("Custom-Made case", Lx=2.0e7, Ly=1.0e7, boundary_condition='allwalls')
​
```
​<br>
Feel free to adjust the content and formatting according to your specific needs and preferences.
