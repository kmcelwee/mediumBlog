import time, math, random

def norm(point, norm_x, norm_y, min_x, min_y):
	return [(point[0] - min_x) * norm_x, (point[1] - min_y) * norm_y]

def draw_tour(tour):
	from graphics import Point, Circle, Line, GraphWin
	maxX = max([x for x, y in tour])
	minX = min([x for x, y in tour])
	maxY = max([y for x, y in tour])
	minY = min([y for x, y in tour])

	N = 750
	norm_x = N/(maxX - minX)
	norm_y = N/(maxY - minY)

	win = GraphWin("TSP", N, N)
	for n in tour:
		c = Point(*norm(n, norm_x, norm_y, minX, minY))
		c = Circle(c, 2)
		c.draw(win)

	for i, _ in enumerate(tour[:-1]):	
		p1 = norm(tour[i], norm_x, norm_y, minX, minY)
		p2 = norm(tour[i+1], norm_x, norm_y, minX, minY)
		l = Line(Point(*p1), Point(*p2))
		l.draw(win)

	win.getMouse()
	win.close()

def distance(p1, p2):
	'''return the distance between two points'''
	return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def tour_length(tour):
	return sum([distance(tour[i], tour[i+1]) for i, _ in enumerate(tour[:-1])])

def nearest_neighbor(all_nodes):
	tic = time.time()
	tour = [all_nodes[0], all_nodes[0]]
	for n in all_nodes[1:]:
		shortest = float('inf')
		index = None
		for i, _ in enumerate(tour[:-1]):
			new_tour = list(tour)
			new_tour.insert(i+1, n)
			dist = tour_length(new_tour)
			if dist < shortest:
				index = i+1
				shortest = dist
		tour.insert(index, n)
		#draw_tour(tour)
	#return tour + [all_nodes[0]]
	return time.time() - tic, tour_length(tour + [all_nodes[0]])

def main():
	N = 4000
	tic = time.time()
	all_nodes = [[random.random(), random.random()] for _ in range(N)]
	tour = nearest_neighbor(all_nodes)
	print(tour_length(tour), time.time() - tic)
	draw_tour(tour)

if __name__ == '__main__':
	main()