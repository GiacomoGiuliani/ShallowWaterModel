# Import required libraries to run the model
import numpy as np 
import matplotlib.pyplot as plt
import time
from time import sleep
from IPython import display
from __future__ import (print_function, division)
from pickle import FALSE

class ShallowWaterModel:
    def __init__(self, case, nx=None, ny=None, dt=None, save_interval=None, f0=None, H=None, gx=None, gy=None, 
                nu=None, OME=None, a=None, g=None, beta_effect=True, Lx=None, Ly=None, nsteps=None, experiment=None, 
                phi=None, gr=None, r=None, boundary_condition=None):
        
        
        self.common_params = {
            "nx": 128,
            "ny": 129,
            "dt": 1000,
            "save_interval": 10,
            "f0": 1e-4,
            "H": 100,
            "gx": 2.0e6,
            "gy": 0,
            "nu": 1e-6,
            "OME": 7.3e-5, 
            "a":6371e3, 
            "g":9.81
        }
        
        self.case_params = {
    "Non-Rotational Waves": {
        "beta_effect": False,
        "Lx": 2.0e7,
        "Ly": 2.0e7,
        "nsteps": 1000,
        "experiment": '2d',
        "phi": 0,
        "gr": 200e3,
        "r": 1e-4,
        "boundary_condition": 'periodic'
    },
    "Rotational Waves (f)": {
        "beta_effect": False,
        "Lx": 2.0e7,
        "Ly": 2.0e7,
        "nsteps": 1200,
        "experiment": '2d',
        "phi": 20,
        "gr": 200e3,
        "r": 1e-4,
        "boundary_condition": 'periodic'
    },
    "Rotational Waves (beta)": {
        "beta_effect": True,
        "Lx": 2.0e7,
        "Ly": 2.0e7,
        "nsteps": 2000,
        "experiment": '2d',
        "phi": 20,
        "gr": 200e3,
        "r": 1e-4,
        "boundary_condition": 'periodic'
    },
    "Equatorially-trapped waves (s)": {
        "beta_effect": True,
        "Lx": 2.0e7,
        "Ly": 1.5e7,
        "nsteps": 1200,
        "experiment": '2d',
        "phi": 0,
        "gr": 200e3,
        "r": 1e-4,
        "boundary_condition": 'periodic'
    },
    "Equatorially-trapped waves (l)": {
        "beta_effect": True,
        "Lx": 2.0e7,
        "Ly": 1.5e7,
        "nsteps": 1000,
        "experiment": '2d',
        "phi": 0,
        "gr": 1e6,
        "r": 1e-4,
        "boundary_condition": 'periodic'
    },
    "Equatorially-trapped waves (a)": {
        "beta_effect": True,
        "Lx": 2.0e7,
        "Ly": 1.5e7,
        "nsteps": 2000,
        "experiment": '2d',
        "phi": 0,
        "gr": 1e6,
        "r": 1e-4,
        "boundary_condition": 'allwalls'
    },
    "Custom-Made case": {
        "nx": nx,
        "ny": ny,
        "dt": dt,
        "save_interval": save_interval,
        "f0": f0,
        "H": H,
        "gx": gx,
        "gy": gy,
        "nu": nu,
        "OME": OME, 
        "a": a, 
        "g": g, 
        "beta_effect": beta_effect,
        "Lx": Lx,
        "Ly": Ly,
        "nsteps": nsteps,
        "experiment": experiment,
        "phi": phi,
        "gr": gr,
        "r": r,
        "boundary_condition": boundary_condition

    },      
}

        if case not in self.case_params:
            raise ValueError("Invalid case specified.")

        # Initialize the model with parameters based on the chosen case
        self.initialize_model(case)
        
        # Case available globally 
        self.case = case

    def initialize_model(self, case):        
        common_params = self.common_params
        case_params = self.case_params[case]
        def_params = self.case_params["Non-Rotational Waves"]
        
        parameters = ["nx", "ny", "Lx", "Ly", "dt", "nsteps", "save_interval", 
                      "experiment", "phi", "f0", "OME", "a", "beta_effect", 
                      "H", "g", "gx", "gy", "gr", "nu", "r", "boundary_condition"]
        
        # Determine which params to use based on the case
        if case != "Custom-Made case":
            params = {**common_params, **case_params}  
            
            for param in parameters:
                setattr(self, param, params.get(param, common_params.get(param)))
                
                
        elif case == "Custom-Made case":
            params = case_params
            default_params = {**common_params, **def_params} 
            
            for param in parameters:
                
                # If the parameter hasn't been defined, the model automatically assigns the default value
                # (Default case = Non Rotational Waves)
                if params.get(param)==None:
                    setattr(self, param, default_params.get(param, common_params.get(param)))
                    print(param)
                
                else: 
                    setattr(self, param, params.get(param, common_params.get(param)))
                    
     
    ## GRID
    # Setup the Arakawa-C Grid:
    #
    # +-- v --+
    # |       |    * (nx, ny)   h points at grid centres
    # u   h   u    * (nx+1, ny) u points on vertical edges  (u[0] and u[nx] are boundary values)
    # |       |    * (nx, ny+1) v points on horizontal edges
    # +-- v --+
    #
    # Variables preceeded with underscore  (_u, _v, _h) include the boundary values,
    # variables without (u, v, h) are a view onto only the values defined within the domain

    def initialize_grid(self):
        nx = self.nx
        ny = self.ny
        Lx = self.Lx
        Ly = self.Ly
        
        self._u = np.zeros((nx+3, ny+2))
        self._v = np.zeros((nx+2, ny+3))
        self._h = np.zeros((nx+2, ny+2))

        self.u = self._u[1:-1, 1:-1]               # (nx+1, ny)
        self.v = self._v[1:-1, 1:-1]               # (nx, ny+1)
        self.h = self._h[1:-1, 1:-1]               # (nx, ny)

        self.state = np.array([self.u, self.v, self.h], dtype='object')
        
        # Initialisation of n-1, n-2 states
        self._pdstate = 0
        self._ppdstate = 0
        
        
        # Size of the grid
        self.dx = Lx / nx            # [m]
        self.dy = Ly / ny            # [m]

        # positions of the value points in [m]
        self.ux = (-Lx/2 + np.arange(nx+1)*self.dx)[:, np.newaxis]
        self.vx = (-Lx/2 + self.dx/2.0 + np.arange(nx)*self.dx)[:, np.newaxis]

        self.vy = (-Ly/2 + np.arange(ny+1)*self.dy)[np.newaxis, :]
        self.uy = (-Ly/2 + self.dy/2.0 + np.arange(ny)*self.dy)[np.newaxis, :]

        self.hx = self.vx
        self.hy = self.uy

        self.t = 0.0                 # [s] Time since start of simulation
        self.tc = 0                  # [1] Number of integration steps taken

        
    def set_initialconditions(self):
        # Set the initial state of the model here by assigning to u[:], v[:] and h[:].
        v0 = self.v * 0.0
        u0 = self.u * 0.0
        
        if self.experiment == '2d':
            # create a single disturbance in the domain:
            # a gaussian at position gx, gy, with radius gr
            # gx =  2.0e6
            # gy =  0.0
            # gr =  2.0e5
            h0 = np.exp(-((self.hx - self.gx)**2 + (self.hy - self.gy)**2)/(2*self.gr**2))*self.H*0.01
        
        if self.experiment == '1d':
            h0 = -np.tanh(100*self.hx/self.Lx)
            # no damping in y direction
            self.r = 0.0

        # set the variable fields to the initial conditions
        self.u[:] = u0
        self.v[:] = v0
        self.h[:] = h0
        
    def useful_quantities(self):
        self.R = self.a  *  np.cos(np.deg2rad(self.phi))
        self.f0 = 2 * self.OME * np.sin(np.deg2rad(self.phi))
        self.U = np.sqrt(self.g * self.H)
        self.Ld_e = 0
        
        if self.beta_effect:
            self.beta =  (2 * self.OME * np.cos(np.deg2rad(self.phi))) / self.R
            self.Ld_e = np.sqrt(self.U/self.beta)

        
    def update_boundaries(self):
        _u = self._u
        _v = self._v
        _h = self._h
        boundary_condition = self.boundary_condition

        # 1. Periodic Boundaries
        #    - Flow cycles from left-right-left
        #    - u[0] == u[nx]
        if boundary_condition == 'periodic':
            _u[0, :] = _u[-3, :]
            _u[1, :] = _u[-2, :]
            _u[-1, :] = _u[2, :]
            _v[0, :] = _v[-2, :]
            _v[-1, :] = _v[1, :]
            _h[0, :] = _h[-2, :]
            _h[-1, :] = _h[1, :]
        # This applied for both boundary cases above
            for field in self.state:
                # Free-slip of all variables at the top and bottom
                field[:, 0] = field[:, 1]
                field[:, -1] = field[:, -2]
                # fix corners to be average of neighbours
                field[0, 0] =  0.5*(field[1, 0] + field[0, 1])
                field[-1, 0] = 0.5*(field[-2, 0] + field[-1, 1])
                field[0, -1] = 0.5*(field[1, -1] + field[0, -2])
                field[-1, -1] = 0.5*(field[-1, -2] + field[-2, -1])

        # 2. Solid walls left and right
        #    - No zonal (u) flow through the left and right walls
        #    - Zero x-derivative in v and h
        if boundary_condition == 'LRwalls':
            # No flow through the boundary at x=0
            _u[0, :] = 0
            _u[1, :] = 0
            _u[-1, :] = 0
            _u[-2, :] = 0

            # free-slip of other variables: zero-derivative
            _v[0, :] = _v[1, :]
            _v[-1, :] = _v[-2, :]
            _h[0, :] = _h[1, :]
            _h[-1, :] = _h[-2, :]
            # This applied for both boundary cases above
            for field in state:
                # Free-slip of all variables at the top and bottom
                field[:, 0] = field[:, 1]
                field[:, -1] = field[:, -2]
                # fix corners to be average of neighbours
                field[0, 0] =  0.5*(field[1, 0] + field[0, 1])
                field[-1, 0] = 0.5*(field[-2, 0] + field[-1, 1])
                field[0, -1] = 0.5*(field[1, -1] + field[0, -2])
                field[-1, -1] = 0.5*(field[-1, -2] + field[-2, -1])

        # 3. Solid walls left, right, top, and bottom
        #    - No zonal (u) flow through the left and right walls
        #    - No meridional (v) flow through the top and bottom walls
        if boundary_condition == 'allwalls':
            # No flow through the boundary at x=0
            _u[0, :] = 0
            _u[1, :] = 0
            _u[-1, :] = 0
            _u[-2, :] = 0

            # # free-slip of other variables: zero-derivative
            # _v[0, :] = _v[1, :]
            # _v[-1, :] = _v[-2, :]
            # _h[0, :] = _h[1, :]
            # _h[-1, :] = _h[-2, :]

            # No flow through the boundary at y=0
            _v[:, 0] = 0
            _v[:, 1] = 0
            _v[:, -1] = 0
            _v[:, -2] = 0

            # # free-slip of other variables: zero-derivative
            # _u[:, 0] = _u[:, 1]
            # _u[-1, :] = _u[:, -2]
            # _h[:, 0] = _h[:, 1]
            # _h[:, -1] = _h[:, -2]
    

    def diffx(self, psi):
        """Calculate ∂/∂x[psi] over a single grid square.

        i.e. d/dx(psi)[i,j] = (psi[i+1/2, j] - psi[i-1/2, j]) / dx

        The derivative is returned at x points at the midpoint between
        x points of the input array."""
        return (psi[1:,:] - psi[:-1,:]) / self.dx

    def diff2x(self, psi):
        """Calculate ∂2/∂x2[psi] over a single grid square.

        i.e. d2/dx2(psi)[i,j] = (psi[i+1, j] - psi[i, j] + psi[i-1, j]) / dx^2

        The derivative is returned at the same x points as the
        x points of the input array, with dimension (nx-2, ny)."""
        return (psi[:-2, :] - 2*psi[1:-1, :] + psi[2:, :]) / self.dx**2

    def diff2y(self, psi):
        """Calculate ∂2/∂y2[psi] over a single grid square.

        i.e. d2/dy2(psi)[i,j] = (psi[i, j+1] - psi[i, j] + psi[i, j-1]) / dy^2

        The derivative is returned at the same y points as the
        y points of the input array, with dimension (nx, ny-2)."""
        return (psi[:, :-2] - 2*psi[:, 1:-1] + psi[:, 2:]) / self.dy**2

    def diffy(self, psi):
        """Calculate ∂/∂y[psi] over a single grid square.

        i.e. d/dy(psi)[i,j] = (psi[i, j+1/2] - psi[i, j-1/2]) / dy

        The derivative is returned at y points at the midpoint between
        y points of the input array."""
        return (psi[:, 1:] - psi[:,:-1]) / self.dy

    def centre_average(self, psi):
        """Returns the four-point average at the centres between grid points."""
        return 0.25*(psi[:-1,:-1] + psi[:-1,1:] + psi[1:, :-1] + psi[1:,1:])

    def y_average(self,psi):
        """Average adjacent values in the y dimension.
        If psi has shape (nx, ny), returns an array of shape (nx, ny - 1)."""
        return 0.5*(psi[:,:-1] + psi[:,1:])

    def x_average(self,psi):
        """Average adjacent values in the x dimension.
        If psi has shape (nx, ny), returns an array of shape (nx - 1, ny)."""
        return 0.5*(psi[:-1,:] + psi[1:,:])
    
    def divergence(self):
        """Returns the horizontal divergence at h points."""
        return self.diffx(self.u) + self.diffy(self.v)

    def del2(self, phi):
        """Returns the Laplacian of self.psi."""
        return self.diff2x(phi)[:, 1:-1] + self.diff2y(phi)[1:-1, :]

    def uvatuv(self):
        """Calculate the value of u at v and v at u."""
        ubar = self.centre_average(self._u)[1:-1, :]
        vbar = self.centre_average(self._v)[:, 1:-1]
        return ubar, vbar

    def uvath(self):
        """Calculate the value of u at h and v at h."""
        ubar = self.x_average(self.u)
        vbar = self.y_average(self.v)
        return ubar, vbar

    def absmax(self, psi):
        """Calculate the absolute maximum value of psi."""
        return np.max(np.abs(psi))

## DYNAMICS
# These functions calculate the dynamics of the system we are interested in

    def forcing(self):
        """Add some external forcing terms to the u, v and h equations.
        This function should return a state array (du, dv, dh) that will
        be added to the RHS of equations (1), (2) and (3) when
        they are numerically integrated."""
    
        du = np.zeros_like(self.u)
        dv = np.zeros_like(self.v)
        dh = np.zeros_like(self.h)
        # Calculate some forcing terms here...
        return np.array([du, dv, dh],dtype=object)
    
    def create_sponge(self):
        """ The purpose of the sponge layer is to absorb outgoing waves
        and dampen oscillations near the boundaries, thus preventing 
        reflection of waves back into the domain. """
        sponge_ny = self.ny//7
        sponge = np.exp(-np.linspace(0, 5, sponge_ny))
        return sponge_ny, sponge
    
    def damping(self, var):
    # sponges are active at the top and bottom of the domain by applying Rayleigh friction
    # with exponential decay towards the centre of the domain
        sponge_ny, sponge = self.create_sponge()
        var_sponge = np.zeros_like(var)
        var_sponge[:, :sponge_ny] = sponge[np.newaxis, :]
        var_sponge[:, -sponge_ny:] = sponge[np.newaxis, ::-1]
        return var_sponge*var
    
    def rhs(self):
        """Calculate the right hand side of the u, v, and h equations."""
        # Unpack common parameters for readability
        f0 = self.f0
        if not self.beta_effect:
            self.beta = 0
        beta = self.beta
        nu = self.nu
        r = self.r
        
        # Calculate u at v and v at u
        u_at_v, v_at_u = self.uvatuv()   # (nx, ny+1), (nx+1, ny)

        # Calculate height equation 
        h_rhs = -self.H*self.divergence() + nu*self.del2(self._h) - r*self.damping(self.h)

        # Calculate u equation 
        dhdx = self.diffx(self._h)[:, 1:-1]  # (nx+1, ny)
        u_rhs = (f0 + beta*self.uy)*v_at_u - self.g*dhdx + nu*self.del2(self._u) - r*self.damping(self.u)

        # Calculate v equation 
        dhdy  = self.diffy(self._h)[1:-1, :]   # (nx, ny+1)
        v_rhs = -(f0 + beta*self.vy)*u_at_v - self.g*dhdy + nu*self.del2(self._v) - r*self.damping(self.v)
        
        return np.array([u_rhs, v_rhs, h_rhs],dtype=object) + self.forcing()

    def step(self):
        dt = self.dt
    
        self.update_boundaries()
        dstate = self.rhs()

        # Take Adams-Bashforth step in time
        if self.tc == 0:
            # Forward Euler
            dt1 = dt
            dt2 = 0.0
            dt3 = 0.0
        elif self.tc == 1:
            # AB2 at step 2
            dt1 = 1.5 * dt
            dt2 = -0.5 * dt
            dt3 = 0.0
        else:
            # AB3 from step 3 on
            dt1 = 23. / 12. * dt
            dt2 = -16. / 12. * dt
            dt3 = 5. / 12. * dt

        newstate = self.state + dt1 * dstate + dt2 * self._pdstate + dt3 * self._ppdstate
        self.u[:], self.v[:], self.h[:] = newstate
        self._ppdstate = self._pdstate
        self._pdstate = dstate

        self.t += dt
        self.tc += 1
        
        
## PLOT
# Basic plots to show the evolution of the simulation at any given time-step
    def plot_all(self, u, v, h, t):
        nc = 12
        colorlevels = np.concatenate([np.linspace(-1, -.05, nc), np.linspace(.05, 1, nc)])
        hmax = np.max(np.abs(h))
        plt.clf()
        plt.subplot(222)
        X, Y = np.meshgrid(self.ux, self.uy)
        plt.contourf(X/self.Lx, Y/self.Ly, u.T, cmap=plt.cm.RdBu_r, levels=colorlevels*self.absmax(u))
        #plt.colorbar()
        plt.grid()
        plt.title('u')

        plt.subplot(224)
        X, Y = np.meshgrid(self.vx, self.vy)
        plt.contourf(X/self.Lx, Y/self.Ly, v.T, cmap=plt.cm.RdBu_r, levels=colorlevels*self.absmax(v))
        #plt.colorbar()
        plt.grid()
        plt.title('v')

        plt.subplot(221)
        X, Y = np.meshgrid(self.hx, self.hy)
        plt.contourf(X/self.Lx, Y/self.Ly, h.T, cmap=plt.cm.RdBu_r, levels=colorlevels*self.absmax(h))
        #plt.colorbar()
        plt.grid()
        if self.t is None:
            plt.title('h')
        else:
            plt.title('h' + '  time=' + str(round(t, 5)))

        plt.subplot(223)
        plt.plot(self.hx/self.Lx, h[:, self.ny//2])
        plt.xlim(-0.5, 0.5)
        plt.ylim(-self.absmax(h), self.absmax(h))
        plt.ylabel('h along x=0')
        plt.grid()
        plt.pause(0.001)
        plt.draw()
        
        
    def Hovmuller(self, y=65, amp=0.1, cmap='seismic', fs = 11):
        # h(x,t) at fixed y (since ny = 129, as the disturbance is generated in
        # centre of the domain, y = 65)
        
        # amp is the range of the colorscale, suggested between 0.05 and 0.8 depending on the case
        # cmap is the colormap, choose the one you like more
        # fs is the fontsize
        
        # Define the figure
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,8))
        pcm = ax.pcolor(self.ha[:,y,:].T,cmap=cmap, vmin=-amp, vmax=amp)
        ax.set_xlabel('x-index', fontsize=fs)
        ax.set_ylabel('time', fontsize=fs)
        ax.set_title('Hovmöller Diagram', fontsize=fs+5)
        ax.grid()
        cbar = fig.colorbar(pcm, ax=ax, orientation='vertical', extend='both')
        plt.show()

    
    def anim_simulation(self, step = 2):
        plt.axis([0, 10, 0, 10])
        for t in range(0,len(self.ta)-1,step):
            self.plot_all(self.ua[:,:,t], self.va[:,:,t], self.ha[:,:,t],self.ta[t])
            display.display(plt.gcf())
            display.clear_output(wait=True)
            sleep(0.05)
         
        # Keep the final frame
        self.plot_all(self.ua[:,:,-1], self.va[:,:,-1], self.ha[:,:,-1], self.ta[-1])
        display.display(plt.gcf())
        


## RUN INFORMATION
# Provide the simulation information
    def print_simulation_info(self):
        
        print("")
        print("")
        print("")
        
        print('*** RUNNING CASE: ' + self.case + ' ***')
        print('Reference Latitude: ' + str(self.phi))
        print(' >> f0: ' + str(round(self.f0, 10)) + ' [s^-1]')
        print(' >> beta: ' + str(round(self.beta, 12)) + ' [m^-1.s^-1]')
        if self.f0 == 0:
            print('Rossby deformation radius, Ld: infinite')
            print('Equatorial Rossby deformation radius, Ld: ' + str(round(self.Ld_e / 1000)) + ' km')
        else:
            Ld = np.sqrt(self.g * self.H) / self.f0
            print('Rossby deformation radius, Ld: ' + str(round(Ld / 1000)) + ' km')
        print('Domain geometry')
        print(' >>Average depth H: ' + str(self.H) + ' m')
        print(' >>Longitudinal extension Lx: ' + str(self.Lx / 1000) + ' km')
        print(' >>Latitudinal extension Ly: ' + str(self.Ly / 1000) + ' km')
        print('IC perturbation in the sea surface elevation:')
        print(' >>position x: ' + str(self.gx / 1000) + ' km')
        print(' >>position y: ' + str(self.gy / 1000) + ' km')
        print(' >>radius: ' + str(self.gr / 1000) + ' km')
        print('Phase speed gravity waves including kelvin waves - sqrt(g * H):')
        print(' >> C=' + str(round(self.U, 2)) + ' m/s')
        print(' >> C=' + str(round(self.U * 3.6, 2)) + ' km/h')
        CFL = self.U * self.dt / (self.Lx / self.nx)
        print(' ')
        print('CFL num. stability criteria: ' + str(round(CFL, 2)))
        print(' ')
        print('Lenght: ' + str(round(self.dt * self.nsteps / 86400, 2)) + ' days')
        print('time step increment, dt: ' + str(self.dt) + 's')
        print('total # of time steps run: ' + str(self.nsteps))
        print('Saving frequency in time steps: ' + str(self.save_interval))
        print('total # of time steps run saved: ' + str(int(self.nsteps / self.save_interval)))
        print('Saving frequency in days: ' + str(round(self.dt * self.save_interval / 86400, 2)) + ' days')

        


## RUN
    def model_run(self):
        # Create the grid
        self.initialize_grid()
        
        # Set initial conditions
        self.set_initialconditions()
        
        # Calculate some useful parameters
        self.useful_quantities()
        
        
         # Define the field variables
        self.ua = np.empty((self.nx + 1, self.ny, int(self.nsteps / self.save_interval)))
        self.va = np.empty((self.nx, self.ny + 1, int(self.nsteps / self.save_interval)))
        self.ha = np.empty((self.nx, self.ny, int(self.nsteps / self.save_interval)))
        self.ta = np.zeros(int(self.nsteps / self.save_interval))  # time
        
        # The simulation, finally!
        j = 0
        for i in range(self.nsteps):
            self.step()  # Assuming step is a method defined within the class
            if i % self.save_interval == 0:
                self.ta[j] = self.t / 86400
                self.ua[:, :, j] = self.u
                self.va[:, :, j] = self.v
                self.ha[:, :, j] = self.h
                print('t [days]  u,v [m/s]   h [m]')
                print('[t={:7.2f} (days)  u: [{:.3f}, {:.3f}] (m/s), v: [{:.3f}, {:.3f}] (m/s), h: [{:.3f}, {:.2f}] (m)'.format(
                    self.t / 86400,
                    self.u.min(), self.u.max(),
                    self.v.min(), self.v.max(),
                    self.h.min(), self.h.max()))
                j += 1
                
        # Print Run information
        self.print_simulation_info()

    def download_arrays(self):
        return self.ua, self.va, self.ha, self.ta
        
       
