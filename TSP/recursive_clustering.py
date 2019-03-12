import numpy as np
import random, time, itertools, math, csv
from sklearn.cluster import KMeans

# ------------------------- DRAWING FUNCTIONS ------------------------------- #

def norm(point, norm_x, norm_y, min_x, min_y):
	return [(point[0] - min_x) * norm_x, (point[1] - min_y) * norm_y]

def draw_tour(path_d, all_nodes):
	from graphics import Point, Circle, Line, GraphWin
	maxX = max([x for x, y in all_nodes])
	minX = min([x for x, y in all_nodes])
	maxY = max([y for x, y in all_nodes])
	minY = min([y for x, y in all_nodes])

	N = 750
	norm_x = N/(maxX - minX)
	norm_y = N/(maxY - minY)

	win = GraphWin("TSP", N, N)
	for n in all_nodes:
		c = Point(*norm(n, norm_x, norm_y, minX, minY))
		c = Circle(c, 3)
		c.draw(win)

	for (k, subdict) in path_d.iteritems():
		#t = Text(Point(*subdict['center']*N), k)
		#t.draw(win)
		if not subdict['hasBeenSplit']:
			p = Point(*norm(subdict['center'], norm_x, norm_y, minX, minY))

			c1i, c2i = subdict['connections']
			c1 = Point(*norm(path_d[str(c1i)]['center'], norm_x, norm_y, minX, minY))
			c2 = Point(*norm(path_d[str(c2i)]['center'], norm_x, norm_y, minX, minY))
			l1 = Line(p, c1)
			l2 = Line(p, c2)
			l1.draw(win)
			l2.draw(win)

	win.getMouse()
	win.close()

# ----------------------------- HELPER FUNCTIONS ----------------------------- #

def distance(p1, p2):
	'''return the distance between two points'''
	return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def path_distance(a, path):
	'''given an array of points and a path, return the length of that path'''
	return sum([distance(a[path[i]], a[path[i+1]]) for i, _ in enumerate(path[:-2])])

def connection_dict(path):
	'''return the two points that each node is connected to given a path'''
	return {str(p): [path[i-1], path[i+1]] for i, p in enumerate(path[:-2])}

def tour_length(path_d):
	'''given the dictionary of points, what's the length of the tour'''
	start_and_finish = len(path_d) - 1
	last_node = None
	this_node = start_and_finish
	next_node = path_d[str(this_node)]['connections'][0] # arbitrary direction
	length = distance(path_d[str(this_node)]['center'], path_d[str(next_node)]['center'])
	
	# follow the connections until we get back to the beginning
	while next_node != start_and_finish:
		last_node = this_node
		this_node = next_node
		cs = list(path_d[str(this_node)]['connections']) # HACK: copy of list bc python 2
		cs.pop(cs.index(last_node))
		next_node = cs[0]
		length += distance(path_d[str(this_node)]['center'], path_d[str(next_node)]['center'])
	return length

def create_connection_dictionary(path_d, clusters, c0i, c1i):
	'''given the path dictionary, the new clusters, and the connections to the 
	existing tour, find the shortest permutation of all points. Return a 
	dictionary that lists the connections between all new points'''
	count = len(path_d)
	small_d = {
		str(c0i): path_d[str(c0i)]['center'],
		str(c1i): path_d[str(c1i)]['center']
	}

	for i, c in enumerate(clusters):
		# add count because one of the connectors might be within 0-3
		small_d[str(i+count)] = c 
	
	combos = itertools.permutations(range(count, count+len(clusters)), len(clusters))
	combos = [[c0i] + list(i) + [c1i] for i in combos]

	shortest = float('inf')
	shortest_p = None
	for path in combos:
		dist = sum([distance(small_d[str(path[i])], small_d[str(path[i+1])]) for i, _ in enumerate(path[:-1])])
		if dist < shortest:
			shortest_p = path
			shortest = dist

	connections = {} 
	for i, p in enumerate(shortest_p):
		if i not in [0, len(shortest_p)-1]:
			connections[str(p-count)] = [shortest_p[i+1], shortest_p[i-1]]

	return connections, shortest_p[1]-count, shortest_p[-2]-count

def initialize_tour(nodes, K):
	'''Find first clusters, and a make complete tour'''
	path_d = {}

	kmeans_s = KMeans(n_clusters=K, random_state=0).fit(nodes)
	clusters = kmeans_s.cluster_centers_

	subnodes = {}
	for i, p in zip(kmeans_s.labels_, nodes):
		subnodes[str(i)] = subnodes.get(str(i), []) + [p]

	# an ugly implementation of all possible paths.
	# there is a prettier and more efficient way of doing this
	all_paths = itertools.permutations(range(1, K), K-1)
	all_paths = [[0] + list(i) + [0] + [list(i)[-1]] for i in all_paths]

	path_distances = [path_distance(clusters, path) for path in all_paths]
	best_path = all_paths[path_distances.index(min(path_distances))]
	connections = connection_dict(best_path)

	for i, c in enumerate(clusters):
		path_d[str(i)] = {
			'center': c,
			'connections': connections[str(i)],
			'subnodes': subnodes[str(i)],
			'hasBeenSplit': False
		}
	return path_d

def split_node(path_d, node_being_split, K):
	'''Given a cluster center, implement KMeans clustering on its subnodes if 
	the number of subnodes is greater than K, and integrate new clusters or 
	points into the existing tour.'''

	subdict = path_d[str(node_being_split)]
	nodes = subdict['subnodes']
	c0i, c1i = subdict['connections']
	count = len(path_d)

	if len(nodes) > K:
		kmeans_s = KMeans(n_clusters=K, random_state=0).fit(nodes)
		clusters = kmeans_s.cluster_centers_

		subnodes = {}
		for i, p in zip(kmeans_s.labels_, nodes):
			subnodes[str(i)] = subnodes.get(str(i), []) + [p]

		connections, p_out0_i, p_out1_i = create_connection_dictionary(path_d, clusters, c0i, c1i)
	else:
		clusters = nodes
		subnodes = {str(i): [p] for i, p in enumerate(nodes)}
		connections, p_out0_i, p_out1_i = create_connection_dictionary(path_d, clusters, c0i, c1i)

	for i, c in enumerate(clusters):	
		path_d[str(i + count)] = {
			'center': c,
			'connections': connections[str(i)],
			'subnodes': subnodes[str(i)],
			'hasBeenSplit': False
		}

	# adjust connector nodes
	path_d[str(c1i)]['connections'][path_d[str(c1i)]['connections'].index(node_being_split)] = p_out1_i+count
	path_d[str(c0i)]['connections'][path_d[str(c0i)]['connections'].index(node_being_split)] = p_out0_i+count

	path_d[str(node_being_split)]['hasBeenSplit'] = True
	return path_d

def implement_recursive_clustering(N, K, all_nodes): # remove all nodes
	'''split nodes until all clusters converge onto the solution'''
	path_d = initialize_tour(all_nodes, K)
	single_nodes = 0
	m = 0
	while single_nodes < N:
		if len(path_d[str(m)]['subnodes']) == 1:
			single_nodes += 1
		else:
			if not path_d[str(m)]['hasBeenSplit']:
				#draw(path_d, all_nodes, saveFile='all_images/'+str(m))
				path_d = split_node(path_d, m, K)
		m += 1
	return path_d

def read_nodes_csv(file):
	'''given a csv file with two columns: X and Y, return a usable array'''
	all_nodes = []
	with open(file) as f:
		for i, row in enumerate(csv.reader(f)):
			all_nodes.append(np.array([float(row[0]), float(row[1])]))
	N = len(all_nodes)
	return all_nodes, N

def print_results(N, K, wait, path_d):
	print('K = {}'.format(K))
	print('run time for {} nodes = {}s'.format(N, wait))
	print('tour distance = {}'.format(tour_length(path_d)))

# ----------------------------- USER FUNCTIONS ------------------------------ #

def solve_file(file, K, draw=True):
	'''given an appropriate CSV, return the solution'''
	all_nodes, N = read_nodes_csv(file)
	tic = time.time()
	path_d = implement_recursive_clustering(N, K, all_nodes)
	wait = time.time() - tic
	#print_results(N, K, wait, path_d)
	if draw:
		draw_tour(path_d, all_nodes)
	return path_d, tour_length(path_d), wait
	
def solve_array(all_nodes, K, draw=True):
	'''given an appropriate array, return the solution'''
	N = len(all_nodes)
	tic = time.time()
	path_d = implement_recursive_clustering(N, K, all_nodes)
	wait = time.time() - tic
	#print_results(N, K, wait, path_d)
	if draw:
		draw_tour(path_d, all_nodes)
	return path_d, tour_length(path_d), wait

def solve_random(N, K, draw=True):
	'''given the number of nodes, create random points and return the solution'''
	all_nodes = [np.array([random.random(), random.random()]) for _ in range(N)]
	tic = time.time()
	path_d = implement_recursive_clustering(N, K, all_nodes)
	wait = time.time() - tic
	#print_results(N, K, wait, path_d)
	if draw:
		draw_tour(path_d, all_nodes)
	return path_d, tour_length(path_d), wait

def main():
	print(solve_random(N=100, K=4))

	# all_nodes = [np.array([random.random(), random.random()]) for _ in range(N)]
	# solve_array(all_nodes, K=4)

	#solve_file('testFiles/usa115475.csv', K=5, draw=False)

if __name__ == '__main__':
	main()
