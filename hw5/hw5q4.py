import pygmsh
import numpy as np
import matplotlib.pyplot as plt

# make the L
h= 0.1
with pygmsh.occ.Geometry() as geom:
    geom.add_polygon(
        [
            [0,0],
            [1,0],
            [1,0.5],
            [0.5,0.5],
            [0.5,1],
            [0,1]
        ],
        h
    )
    mesh_L= geom.generate_mesh()


with pygmsh.occ.Geometry() as geom:
    d1= geom.add_disk([0,0], 2, mesh_size=h)
    d2= geom.add_disk([-1, -3/4], 1/2, mesh_size=h)
    d3= geom.add_disk([1, -3/4], 1/2, mesh_size=h)
    r1= geom.add_polygon([
        [-2, 0],
        [2,0],
        [2,2],
        [-2,2]
    ], mesh_size=h)
    u= geom.boolean_union([d2, d3, r1])
    diff= geom.boolean_difference([d1], u)
    mesh_Holes= geom.generate_mesh()


with pygmsh.occ.Geometry() as geom:
    s= 5
    theta= np.linspace(0, 2 * np.pi, s+1)
    theta= theta[:-1] #we doubled the last point

    #outer
    xs= np.cos(theta)
    ys= np.sin(theta)
    ps_outer= np.stack((xs,ys),axis=1)
    ps_outer= ps_outer.tolist()

    #inner
    phi= 1/2 * (1 + np.sqrt(5))
    xs= (2 - phi) * np.cos(theta + np.pi)
    ys= (2 - phi) * np.sin(theta + np.pi)
    ps_inner= np.stack((xs,ys),axis=1)
    ps_inner= ps_inner.tolist()

    p1= geom.add_polygon(ps_outer, mesh_size=h)
    p2= geom.add_polygon(ps_inner, mesh_size=h)
    d= geom.boolean_difference([p1], [p2])

    mesh_Pent= geom.generate_mesh()



x= mesh_L.points[:,0]
y= mesh_L.points[:,1]
tri= mesh_L.cells_dict['triangle']
plt.scatter(x,y)
plt.triplot(x,y,tri)
plt.gca().set_aspect('equal')
plt.show()

x= mesh_Holes.points[:,0]
y= mesh_Holes.points[:,1]
tri= mesh_Holes.cells_dict['triangle']
plt.scatter(x,y)
plt.triplot(x,y,tri)
plt.gca().set_aspect('equal')
plt.show()

x= mesh_Pent.points[:,0]
y= mesh_Pent.points[:,1]
tri= mesh_Pent.cells_dict['triangle']
plt.scatter(x,y)
plt.triplot(x,y,tri)
plt.gca().set_aspect('equal')
plt.show()