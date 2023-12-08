#call featurizer by "from tile_coding_6d import TileCoder"
#featurizer = TileCoder()
#s = featurizer.featurize(s)

#it should return a one-hot-encoding that represents which tile is a feature present in a particular tiling

#the number of horizontal,vertical,velocity partition are set to 100

#there are 5 tilings total


import numpy as np

class TileCoder:
	def __init__(self):
		self.width = 288
		self.height = 512
		self.velocity = 20

	def create_a_tiling_6d(self, width, height, velocity, numOfPartitions, shifts):
		birdPosition_partition_points = (np.linspace(0,height,numOfPartitions[1])+ shifts[1])[1:-1] 
		velocity_partition_points = (np.linspace(0,velocity,numOfPartitions[2])+ shifts[2])[1:-1]
		pipe1_y_partition_points = (np.linspace(0,height,numOfPartitions[1])+ shifts[1])[1:-1]
		pipe1_x_partition_points = (np.linspace(0,width,numOfPartitions[0])+ shifts[0])[1:-1]
		pipe2_y_partition_points = (np.linspace(0,height,numOfPartitions[1])+ shifts[1])[1:-1]
		pipe2_x_partition_points = (np.linspace(0,width,numOfPartitions[0])+ shifts[0])[1:-1]

		tile = []
		tile.append(birdPosition_partition_points)
		tile.append(velocity_partition_points)
		tile.append(pipe1_y_partition_points)
		tile.append(pipe1_x_partition_points)
		tile.append(pipe2_y_partition_points)
		tile.append(pipe2_x_partition_points)
		return tile

	
	def create_tilings_6d(self):
		#[(#ofHorizontalPartition,#ofVerticalPartition,#ofVelocityPartition),(horizontalShift, verticalShift,velocityShift)]
		tiling_particulars = [
			[(100,100,100),(-0.72*4,-1.29*4,-0.05*4)],
			[(100,100,100),(-0.72*2,-1.29*2,-0.05*2)],
			[(100,100,100),(0.0,0.0,0.0)],
			[(100,100,100),(0.72*2,1.29*2,0.05*2)],
			[(100,100,100),(0.72*4,1.29*4,0.05*4)],
		]
		
		tilings = [self.create_a_tiling_6d(self.width, self.height, self.velocity, numOfPartitions, shifts) for 
		numOfPartitions, shifts in tiling_particulars]
		return tilings


	def featurize_6d(self, state):
		#tilings[0][0] --> is the 1st tile's horizontal partition points
		#tilings[0][1] --> is the 1st tile's vertical partition points
		#feature = [horizontalZone in tile 1, horizontalZone in tile 2, ..., verticalZone in tile 1, verticalZone in tile 2...]
		#or... feature = [horizontalZone in tile 1, verticalZone in tile 1, horizontalZone in tile 2,  verticalZone in tile 2,...]
		bird_y = state[0]
		bird_vel = state[1]
		pipe1_y = state[2]
		pipe1_x = state[3]
		pipe2_y = state[4]
		pipe2_x = state[5]

		tilings = self.create_tilings_6d()

		feature = []

		for i in range(len(tilings)):
			zone = []
			birdPosition_zone = np.digitize(bird_y, tilings[i][0])
			velocity_zone = np.digitize(bird_vel, tilings[i][1])
			pipe1_y_zone = np.digitize(pipe1_y, tilings[i][2])
			pipe1_x_zone = np.digitize(pipe1_x, tilings[i][3])
			pipe2_y_zone = np.digitize(pipe2_y, tilings[i][4])
			pipe2_x_zone  = np.digitize(pipe2_x, tilings[i][5])

			zone.append(birdPosition_zone)
			zone.append(velocity_zone)
			zone.append(pipe1_y_zone)
			zone.append(pipe1_x_zone)
			zone.append(pipe2_y_zone)
			zone.append(pipe2_x_zone)
		
			feature.append(zone)

		#return result in a one-hot-encoding style
		result = []
		for i in range(len(feature)):
			result.append(np.eye(101)[feature[i]])
		flattened_feature = np.array(result).flatten()
		print(flattened_feature)

		return flattened_feature #vector