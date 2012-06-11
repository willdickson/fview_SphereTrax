from __future__ import print_function
import math
import numpy
import scipy.optimize as optimize

# -----------------------------------------------------------------------------
# Routines for finding the 3D position of a sphere relative to camera given 
# the ellipse produced by the boundary of its projection. 
# -----------------------------------------------------------------------------

def find_sphere_pos(z0_min, z0_max, R, fx, fy, cx, cy, coeff_prm,num_pts=50):
    """
    Finds the 3D position of a sphere relative to a camera given the
    coefficients of an ellipse defining the boundary of its projection in a
    camera image plane.

    Arguments:
     z0_min     = minumum z in search range 
     z0_max     = maximum z in search range  
     R          = the spheres radius
     fx, fy,    = camera calibration focal lenghts
     cx, cy     = camera calibration offesets
     coeff_prm  = coefficients for the ellipse produced by the boundry of
                  the projection of the sphere in the image plane. Coefficents
                  should be for parametric form -e.g.  a,b,u0,v0,angle. 

    Returns: x0,y0,z0 position of the sphere relative to the camera.
    """
    pos = estimate_sphere_pos(z0_min, z0_max, R, fx, fy, cx, cy, coeff_prm,num_pts=num_pts)
    pos = refine_sphere_pos(pos, R, fx, fy, cx, cy, coeff_prm, num_pts=num_pts)
    return pos

def estimate_sphere_pos(z0_min, z0_max, R, fx, fy, cx, cy, coeff_prm, num_pts=50):
    """
    Finds an estimate of the 3D position of a sphere relative to a camera given
    the coefficients of an ellipse defining the boundary of its projection in a
    camera image plane.

    Arguments

     z0_min     = minumum z in search range 
     z0_max     = maximum z in search range  
     R          = the spheres radius
     fx, fy,    = camera calibration focal lenghts
     cx, cy     = camera calibration offesets
     coeff_prm  = coefficients for the ellipse produced by the boundry of
                  the projection of the sphere in the image plane. Coefficents
                  should be for parametric form -e.g.  a,b,u0,v0,angle. 

    Returns: estimate of the sphere's position x0,y0,z0  

    Note, this algorithm should be used for producing an intial guess of the
    spheres position which is then feed into a more general (hill climbing)
    optimizer.  It assumes that the center of the ellipse (produced by the
    projection of sphere in the image plane) is on the same ray as center of
    the sphere. This is not in general true. However, for most cameras it is
    close - so this is a useful assumption for producing an initial estimate of
    the sphere position which can then be refined by a more general (hill
    climbing) optimizer. 
    """
    a,b,u0,v0,angle = coeff_prm 
    def cost_func(z0): 
        x0, y0 = get_xy_world_coord(u0,v0,z0,fx,fy,cx,cy)
        test_coeff_imp = get_sphere_proj_bndry_ellipse(x0,y0,z0,R,fx,fy,cx,cy)
        test_coeff_prm = ellipse_implicit_to_parametric(test_coeff_imp)
        x_pts, y_pts = create_ellipse_pts(*test_coeff_prm,num_pts=num_pts)
        dist = mean_distance_to_ellipse(zip(x_pts,y_pts),coeff_prm)
        return dist 

    z0_sol = optimize.fminbound(cost_func,z0_min, z0_max)
    x0_sol, y0_sol = get_xy_world_coord(u0,v0,z0_sol,fx,fy,cx,cy)
    sphere_pos = x0_sol, y0_sol, z0_sol
    sphere_pos = map(float, sphere_pos)
    return sphere_pos

def refine_sphere_pos(pos_init, R, fx, fy, cx, cy, coeff_prm, num_pts=50):
    """
    Use hill climbing optiimizer to refine initial estimate of sphere position.

    Arguments:

     pos_init   = intial estimate of sphere postiion x0,y0,z0
     R          = radius of sphere
     fx,fy      = camera calibration focal lengths
     cx,cy      = camera calibration offsets
     coeff_prm  = coefficients for sphere boundary ellipse in camera image plane 
                  Note, coefficient should specify ellipse in parametric form
                  a,b,u0,v0,angle.

    Return: refined estimate of sphere's position x0,y0,x0
    """
    def cost_func(pos):
        x0,y0,z0 = pos[0],pos[1],pos[2]
        test_coeff_imp = get_sphere_proj_bndry_ellipse(x0,y0,z0,R,fx,fy,cx,cy)
        test_coeff_prm = ellipse_implicit_to_parametric(test_coeff_imp)
        x_pts, y_pts = create_ellipse_pts(*test_coeff_prm,num_pts=num_pts)
        dist = mean_distance_to_ellipse(zip(x_pts,y_pts),coeff_prm)
        return dist 

    pos_sol = optimize.fmin(cost_func,pos_init,disp=False)
    pos_sol = [float(x) for x in pos_sol]
    return tuple(pos_sol)

def get_sphere_proj_bndry_ellipse(x,y,z,R,fx,fy,cx,cy): 
    """
    Get the boundary ellipse of the projection of a sphere in the camera image
    plane.  Returns the coefficients for the implicit equation of the ellipse
    which results from the projection of the sphere, at position (x,y,z) w/
    radius R, into the image plane of a camera with the given parameters, focal
    lengths (fx,fy) and offsets (cx,cy).

    Arguments:

    x,y,z  = position of sphere
    R      = radius of sphere
    fx,fy  = camera calibration focal lengths
    cx,cy  = camera calibration offsets.

    Returns: list of coefficients where 

    coeff[0]*x**2 + coeff[1]*y**2 + coeff[2]*x*y + coeff[3]*x + coeff[4]*y + 1 = 0 

    defines the ellipse.
    """
    # Coeff u**2
    coeff_u2v0 = R**2/fx**2 - y**2/fx**2 - z**2/fx**2

    # Coeff v**2 
    coeff_u0v2 = R**2/fy**2 - x**2/fy**2 - z**2/fy**2

    # Coeff u*v
    coeff_u1v1 = 2*x*y/(fx*fy)

    # Coeff u
    coeff_u1v0 = 2*x*z/fx - 2*cy*x*y/(fx*fy) - 2*cx*R**2/fx**2 
    coeff_u1v0 +=  2*cx*y**2/fx**2 + 2*cx*z**2/fx**2

    # Coeff v 
    coeff_u0v1 = 2*y*z/fy - 2*cx*x*y/(fx*fy) - 2*cy*R**2/fy**2 
    coeff_u0v1 +=  2*cy*x**2/fy**2 + 2*cy*z**2/fy**2

    # Coeff 1
    coeff_u0v0 = -2*cx*x*z/fx - 2*cy*y*z/fy + 2*cx*cy*x*y/(fx*fy) 
    coeff_u0v0 += + R**2*cx**2/fx**2 + R**2*cy**2/fy**2 
    coeff_u0v0 += R**2 - x**2 - y**2 - cx**2*y**2/fx**2 
    coeff_u0v0 -= cx**2*z**2/fx**2 + cy**2*x**2/fy**2 + cy**2*z**2/fy**2

    # Pack coefficients into list and normalize so that coeff_u0v0 is 1.
    coeff_list = [coeff_u2v0, coeff_u0v2, coeff_u1v1, coeff_u1v0, coeff_u0v1, coeff_u0v0]
    coeff_list = [float(x/coeff_list[-1]) for x in coeff_list[:-1]]
    return coeff_list

# -----------------------------------------------------------------------------
# Routines for fitting an ellipse to a set of points in 2D
# -----------------------------------------------------------------------------

def fit_ellipse(pts,constrain_to_circle=False):
    """
    Fits ellipse to the given set of x and y points. Based on the method in 
    "Numerically Stable Direct Least Sqaures Fitting of Ellipses" by Radim Halir
    and Jan Flusser. 

    Arguments:

    pts = list of x, y point paires, [(x0,y0), .... ]

    Return: list of ellipse coefficients in parametric form - a,b,x0,y0,angle

    where

    a,b   = ellipse major and minor axes
    x0,y0 = offset of ellipse center
    angle = rotation angle of ellipse
    """
    if constrain_to_circle:
        x0,y0,R = fit_circle(pts)
        coeff_prm = R, R, x0, y0, 0.0
    else:
        coeff_imp = fit_ellipse_implicit(pts)
        coeff_prm = ellipse_implicit_to_parametric(coeff_imp)
    return coeff_prm

def fit_ellipse_implicit(pts):
    """
    Fits ellipse to the given set of x and y points. Based on the method in 
    "Numerically Stable Direct Least Sqaures Fitting of Ellipses" by Radim Halir
    and Jan Flusser. 

    Arguments:

    pts = list of x, y point paires, [(x0,y0), .... ]

    Return: list of ellipse coefficients in implicit form, i.e., 

    [coeff[0], coeff[1], ..... ] such that

    coeff[0]*x**2 + coeff[1]*y**2 + coeff[2]*x*y + coeff[3]*x + coeff[4]*y + 1 = 0 
    """

    x,y = zip(*pts)
    x = numpy.array(x).reshape((len(x),1))
    y = numpy.array(y).reshape((len(y),1))

    # Quadratic part of design matrix
    D1 = numpy.hstack((x**2, x*y, y**2))

    # Linear part of design matrix
    D2 = numpy.hstack((x, y, numpy.ones(x.shape)))
    
    # Quadratic part of scatter matrix
    S1 = numpy.dot(D1.transpose(), D1)

    # Combined part of scatter matrix
    S2 = numpy.dot(D1.transpose(), D2)

    # Linear part of scatter matrix
    S3 = numpy.dot(D2.transpose(), D2)
    S3_inv = numpy.linalg.inv(S3)

    # Sub-component of constraint matrix
    C1 = numpy.array([[0,0,2],[0, -1, 0], [2, 0, 0]])
    C1_inv = numpy.linalg.inv(C1)

    T = -numpy.dot(S3_inv, S2.transpose())
    M = S1 + numpy.dot(S2,T)
    M = numpy.dot(M,C1_inv)

    eigval, eigvec = numpy.linalg.eig(M)
    cond = 4*eigvec[0,:]*eigvec[2,:] - eigvec[1,:]**2

    a1 = eigvec[:, cond > 0] 
    a = numpy.vstack((a1, numpy.dot(T,a1)))
    a = numpy.ravel(a)
    a = a/a[-1]

    # Re-arrange  coefficients  - so they match those used by the other methods 
    # in this module.
    coeff = [a[0],a[2],a[1],a[3],a[4]]
    return coeff

    
def fit_ellipse_old(pts):
    """
    Finds the best fit ellipse (in the least squared sense) to the given set 
    of points (x,y).

    Arguments:
    pts = [(x0,y0), .... ] list of x,y point pairs

    Returns: fit coefficients in parametric form a,b,x0,y0,angle.
    """
    # Get initial estimate for ellipse parameters using linear fit 
    sol = fit_quadratic(pts)
    params_est = ellipse_implicit_to_parametric(sol)

    # Refine initial estimate using hill climber
    def cost_func(params):
        dist2_list = []
        for x,y in pts:
            dist = distance_to_ellipse(x,y,list(params))
            dist2_list.append(dist**2)
        return sum(dist2_list) 
    params_fit = optimize.fmin(cost_func,params_est,disp=False)
    return params_fit

def fit_quadratic(pts):
    """
    Fits a quadratic form: 

    sol[0]*x**2 + sol[1]*y**2 + sol[2]*x*y + sol[3]*x + sol[4]*y + 1 = 0 

    to the given set of data points, pts = [(x0,y0), (x1,y1), ...] 

    Arguments:
    pts = [(x0,y0), .... ]  list of x,y point pairs. 

    Returns: fit coefficients [sol0, sol1, ...]
    """
    A = numpy.zeros((len(pts),5))
    b = -numpy.ones((len(pts),))
    for i,p in enumerate(pts):
        x,y = p
        A[i,0] = x**2
        A[i,1] = y**2
        A[i,2] = x*y
        A[i,3] = x
        A[i,4] = y
    result = numpy.linalg.lstsq(A,b)
    sol = result[0]
    return sol

def fit_circle(pts):
    """
    Fits a circle (least squares) to the given set of points. 

    Arguments:
    pts = list of point pairs [(x0,y0), ... ]

    Returns: circle parameters 

    x0 = x offset
    y0 = y offset
    R  = radius
    """
    # Extract points and shift by mean
    N = len(pts)
    x,y = zip(*pts)
    x = numpy.array(x)
    y = numpy.array(y)
    u = x - x.mean()
    v = y - y.mean()

    # Create system of linear equations and solve.
    Suv = (u*v).sum()
    Suu = (u*u).sum() 
    Svv = (v*v).sum()
    Suuu = (u*u*u).sum()
    Svvv = (v*v*v).sum()
    Suvv = (u*v*v).sum()
    Svuu = (v*u*u).sum()
    A = numpy.array([[Suu, Suv],[Suv, Svv]])
    b = 0.5*numpy.array([Suuu + Suvv, Svvv + Svuu])
    sol = numpy.linalg.solve(A,b)

    # Shift back to get offset and find radius.
    x0 = sol[0] + x.mean()
    y0 = sol[1] + y.mean()
    R = math.sqrt(sol[0]**2 + sol[1]**2 + (Suu + Svv)/float(N))
    return x0, y0, R


# -----------------------------------------------------------------------------
# Basic ellipse computations
# -----------------------------------------------------------------------------

def distance_to_ellipse(x,y,ellipse_params,tol=1.0e-16,num_sample=50):
    """
    Computes the distance from the point (x,y) to the ellipse given by the
    specified parameters (a,b,x0,y0,angle).

    Arguments:

    x,y            = position of point
    ellipse_params = ellipse parameters/coefficients for the equation of an
                     ellipse in parametric form (a,b,x0,y0,angle)).

    Returns: distance to ellipse.
    """
    a,b,x0,y0,ang = ellipse_params

    done = False
    dist2_min_last = None

    # Set initial sample region
    t0, t1 = 0,2.0*numpy.pi

    while not done:
        # Get sample distances
        t = numpy.linspace(t0,t1,num_sample)
        dt = t[1] - t[0]
        x_ell, y_ell = ellipse(t,a,b,x0,y0,ang)
        dist2 = (x_ell - x)**2 + (y_ell- y)**2

        # Find minimum distance and index
        ind = dist2.argmin()
        t_min = t[ind]
        dist2_min = dist2[ind]

        # Compare with last minimum - if difference is less that tolerance then 
        # we are done. 
        if dist2_min_last is not None:
            change = abs(dist2_min - dist2_min_last)
            if change < tol:
                done = True

        # Update last minimum distance squared and sample region [t0,t1]
        dist2_min_last = dist2_min
        t0 = t[ind] - dt
        t1 = t[ind] + dt
    return numpy.sqrt(dist2_min)

def mean_distance_to_ellipse(pts,ellipse_params,tol=1.0e-16,num_sample=50):
    """
    Computes the mean distance of a set of pts to the ellipse given the coefficients for
    the equation of the ellipse in parametric form.

    Arguments:
    pts = [(x0,y0), ... ] list of point pairs.
    ellipse_params = coefficients defining ellipse in parametric form, a,b,x0,y0,angle

    Returns: mean distance
    """
    dist_list = []
    for x,y in pts:
        dist = distance_to_ellipse(x,y,ellipse_params,tol=tol,num_sample=num_sample)
        dist_list.append(dist)
    dist_array = numpy.array(dist_list)
    return dist_array.mean()

def ellipse_implicit_to_parametric(params):
    """
    Returns the coefficients for the parametric equation of an ellipse

    i.e,  a,b,x0,y0,angle such that 

    x(t) = x0 + a*cos(t)*cos(angle) - b*sin(t)*sin(angle)
    y(t) = y0 + a*cos(t)*sin(anlge) + b*sin(t)*cos(angle)

    given coefficients for the implicit equation of an ellipse, i.e., 
    params[0], ... params[4] such that 

    params[0]*x**2 + params[1]*y**2 + params[2]*x*y + params[3]*x + params[4]*y + 1 = 0 
    
    Arguments:
    params = coefficients defining ellipse in implicit form.

    Returns: coefficients defining ellipse in parametric form.
    """

    # Get matrix form for quadratic equation xt*A*x + Kx + 1 = 0
    A = numpy.array([
        [params[0], 0.5*params[2]],
        [0.5*params[2], params[1]],
        ])
    K = numpy.array([[params[3], params[4]]])

    # Compute eigenvalues and rotation matrix (eigen vectors)
    eig, R = numpy.linalg.eig(A)
    Rt = R.transpose()
    KR = numpy.dot(K,R)

    # Estimate rotation angle required to sqaure the ellipse with the x,y axes
    angle = numpy.arctan2(R[1,0],R[0,0])

    # Estimate the shift required to center the ellipse at the origin
    x0_rotated = -0.5*KR[0,0]/eig[0]
    y0_rotated = -0.5*KR[0,1]/eig[1]
    shift_rotated = numpy.array([[x0_rotated],[y0_rotated]])
    shift = numpy.dot(R,shift_rotated)
    x0 = shift[0,0]
    y0 = shift[1,0]

    # Estimate major and minor axes - a,b
    numer = (KR[0,0]**2)/(4.0*eig[0]) + (KR[0,1]**2)/(4.0*eig[1]) - 1
    a = numpy.sqrt(numer/eig[0])
    b = numpy.sqrt(numer/eig[1])

    return a,b,x0,y0,angle

def ellipse_parametric_to_implicit(params,num_pts=100):
    """
    Returns coefficients for implicit equation of an ellipse 

    sol[0]*x**2 + sol[1]*y**2 + sol[2]*x*y + sol[3]*x + sol[4]*y + 1 = 0 

    given the coefficients for the parametric parametric equation  of
    an ellipse i.e,  a,b,x0,y0,angle such that 

    x(t) = x0 + a*cos(t)*cos(angle) - b*sin(t)*sin(angle)
    y(t) = y0 + a*cos(t)*sin(anlge) + b*sin(t)*cos(angle)

    Arguments:

    params = coefficients defining ellipse in parametric form.

    Returns: coefficients defining ellipse in implicit form.
    """
    a,b,x0,y0,angle = params
    x,y = create_ellipse_pts(a,b,x0,y0,angle,num_pts=num_pts)
    pts = zip(x,y)
    sol = fit_quadratic(pts)
    # --------------------------------------------------------------------------
    # Check fit
    # --------------------------------------------------------------------------
    #for x,y in  pts:
    #    val = sol[0]*x**2 + sol[1]*y**2 + sol[2]*x*y + sol[3]*x + sol[4]*y + 1  
    #    print(x,y,val)
    # --------------------------------------------------------------------------
    return list(sol)

def ellipse(t,a,b,x0,y0,ang):
    """
    Returns x and y point(points) on an ellipse with the given major
    and minor axes (a,b) offset from the origin and rotation angle.

    Arguments:

    t     = parameter or array of parameters [0,2*pi]
    a,b   = major and minor axes of ellipse
    x0,y0 = offset of ellipse
    ang   = rotation angle of ellipse (radians)

    Returns: x,y points on ellipse
    """
    x = x0 + a*numpy.cos(t)*numpy.cos(ang) - b*numpy.sin(t)*numpy.sin(ang)
    y = y0 + a*numpy.cos(t)*numpy.sin(ang) + b*numpy.sin(t)*numpy.cos(ang)
    return x,y

def create_ellipse_pts(a,b,x0,y0,ang,num_pts=100):
    """
    Create (num_pts,) arrays of evenly spaces points on ellipse with major and
    minor axes a and b, rotated by angle ang, shifted to position x0,y0, with
    num_pts points.

    Arguments:

    a,b     = major and minor axes of ellipse
    x0,y0   = offset of ellipse
    ang     = rotation angle of ellipse
    num_pts = number of points in arrays.

    Returns: arrays x,y of evenly space points on ellipse
    """
    t = numpy.linspace(0,2.0*numpy.pi,num_pts)
    x,y = ellipse(t,a,b,x0,y0,ang)
    return x,y

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def rotate_then_shift(x,y,ang,x_shift,y_shift):
    """
    Rotate the (N,) arrays of points x,y by the given angle then shift them by
    the specified amount.

    Arguments:

    x,y = (n,) arrays for points
    ang = rotation angle
    x_shift = x coordinate shift
    y_shift = y coordinate shift

    Return: arrays of rotated and shifted x,y points.
    """
    R = numpy.array([
        [numpy.cos(ang), -numpy.sin(ang)],
        [numpy.sin(ang), numpy.cos(ang)],
        ])
    xy = numpy.array(zip(x,y))
    xy_new = numpy.dot(xy,R.transpose())
    x = [val[0]+x_shift for val in xy_new]
    y = [val[1]+y_shift for val in xy_new]
    return x,y

def random_sample(x,y,num):
    """
    Returns a random sample from the point lists x and y
    
    Arguments:
    x,y  = list or (N,) arrays of points
    num  = number of random samples to return

    Returns: x,y lists of randomly sampled points.
    """
    xy = zip(x,y)
    numpy.random.shuffle(xy)
    xy = xy[:num]
    return [x for x,y in xy], [y for x,y in xy]

def add_noise(x,y,noise_std):
    """
    Adds random noise with given standard devication to the x and y data sets.
    
    Arguments:
    x,y = lists or (N,) arrays of points
    noise_std = noise standard deviation

    Returns: x,y lists of points w/ added noise.
    """
    xx = numpy.array(x)
    yy = numpy.array(y)
    xx = xx + numpy.random.normal(scale=noise_std,size=xx.shape)
    yy = yy + numpy.random.normal(scale=noise_std,size=yy.shape)
    return list(xx), list(yy)

def get_xy_world_coord(u,v,z,fx,fy,cx,cy):
    """
    Gets world frame x,y-coordinates for a point given:

    u,v    = image plane coordinates
    z      = world coordinate z position
    fx,fy  = camera calibration focal lengths
    cx,cy  = camera calibration offsets.

    Returns: world coordinates x,y 
    """
    x = z*(u - cx)/fx
    y = z*(v - cy)/fy
    return x,y

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('GTK')
    import matplotlib.pyplot as plt

    if 1:
        """
        Test sphere finding functions. 
        """

        # Sphere position
        x0 = 0.244 
        y0 = -3.0
        z0 = 126.0
        R = 6.35

        # Camera parameters
        fx = 5478.240997 
        fy = 5478.954144
        cx = 327.500000
        cy = 245.000000

        # Num ellipse fit points
        num_fit_pts = 5 
        constrain_to_circle = True

        # Z search range
        z0_min = 0.1*z0
        z0_max = 1.5*z0

        # Get ellipse defines by boundary of sphere's projection on image plane
        coeff_imp_true = get_sphere_proj_bndry_ellipse(x0,y0,z0,R,fx,fy,cx,cy) 
        coeff_prm_true = ellipse_implicit_to_parametric(coeff_imp_true)
        a,b,u0,v0,angle = coeff_prm_true

        # Draw ellipse
        x_ell, y_ell = create_ellipse_pts(*coeff_prm_true)

        fig1 = plt.figure(1)
        plt.plot(x_ell,y_ell,'b')
        plt.axis('equal')
        plt.title('Enter points')

        # Get sample points for fitting
        vals = plt.ginput(num_fit_pts)
        x_sample = [xx for xx,yy in vals]
        y_sample = [yy for xx,yy in vals]
        pts_sample = zip(x_sample,y_sample)

        coeff_prm_fit = fit_ellipse(pts_sample,constrain_to_circle=constrain_to_circle)
        coeff_imp_fit = ellipse_parametric_to_implicit(coeff_prm_fit)
        x_fit, y_fit = create_ellipse_pts(*coeff_prm_fit)

        pos_fnd = find_sphere_pos(z0_min, z0_max, R, fx, fy, cx, cy, coeff_prm_fit)
        x0_fnd, y0_fnd, z0_fnd = pos_fnd 

        print()
        print('Position Original:')
        print(' x0:', x0)
        print(' y0:', y0)
        print(' z0:', z0)
        print()
        print('Position Found:')
        print(' x0:', x0_fnd)
        print(' y0:', y0_fnd)
        print(' z0:', z0_fnd)

        fig1.clear()
        plt.plot(x_ell,y_ell,'b')
        plt.plot(x_sample,y_sample,'or')
        plt.axis('equal')
        plt.title('Sample Points')
        plt.draw()

        fig2 = plt.figure(2)
        plt.plot(x_ell, y_ell, 'b')
        plt.plot(x_fit, y_fit, 'r')
        plt.title('Boundary Ellipse Fit')
        plt.axis('equal')
        plt.show()

    if 0:
        """
        Test ellipse fitting functions.
        """
        # Sphere position
        x0 = 0.244 
        y0 = -3.0
        z0 = 126.0
        R = 6.35

        # Camera parameters
        fx = 5478.240997 
        fy = 5478.954144
        cx = 327.500000
        cy = 245.000000

        # Number of sample points for fit
        num_samp = 5 
        constrain_to_circle = True

        # Get ellipse defines by boundary of sphere's projection on image plane
        coeff_imp_true = get_sphere_proj_bndry_ellipse(x0,y0,z0,R,fx,fy,cx,cy) 
        coeff_prm_true = ellipse_implicit_to_parametric(coeff_imp_true)
        x,y = create_ellipse_pts(*coeff_prm_true)

        plt.plot(x,y,'b')
        plt.axis('equal')

        if 0:
            x_samp, y_samp = random_sample(x,y,num_samp)
            x_samp, y_samp = add_noise(x_samp,y_samp,5.0)
        else:
            samples = plt.ginput(num_samp)
            x_samp, y_samp = zip(*samples)

        pts_samp = zip(x_samp,y_samp)
        coeff_prm_fit = fit_ellipse(pts_samp,constrain_to_circle=constrain_to_circle)
        x_fit, y_fit = create_ellipse_pts(*coeff_prm_fit)

        plt.plot(x_samp, y_samp, 'or')
        plt.plot(x_fit,y_fit, 'g')
        plt.axis('equal')
        plt.draw()
        plt.show()
         


