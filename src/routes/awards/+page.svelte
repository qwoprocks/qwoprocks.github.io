<svelte:head>
	<title>Awards | Ming Chong Lim</title>
</svelte:head>

<h1>October 2022: Meta Global Hackathon</h1>

<h2>1st Place</h2>

<p>The hackathon took place over 2 weeks, involving the following challenges:</p>

<ul>
	<li>
		Coding Challenge:
		<ul>
			<li>
				Various LeetCode-style questions with difficulties ranging from medium to hard were given,
				with extra points for each question given to the top few fastest participants.
			</li>
		</ul>
	</li>
	<li>
		Linux Challenge:
		<ul>
			<li>Use Linux commands to find secrets inside the filesystem given.</li>
		</ul>
	</li>
	<li>
		Find the Bug Challenge:
		<ul>
			<li>
				Given snippets of programming-language-agnostic code that each executed a certain kind of
				algorithm, find the bug in the code within the time limit.
			</li>
		</ul>
	</li>
	<li>
		Quizzes:
		<ul>
			<li>Quizzes testing general knowledge about programming and Meta.</li>
		</ul>
	</li>
	<li>
		Product Thinking Challenge:
		<ul>
			<li>
				The product thinking challenge involved two stages that could be completed in groups of up
				to 4 people. I completed the challenge in a 2-man team. The first stage involved coming up
				with a 2-slide proposal for a tech idea that could solve a given problem statement, and the
				second stage involved fleshing out the details of our solution, with the best presentation
				chosen by a panel of Meta judges. My team was the overall champion for this challenge.
			</li>
		</ul>
	</li>
</ul>

<h1>August 2021: Citi Hackoverflow</h1>

<h2>1st Place</h2>

<h1>July 2020: BrainHack Today I Learned</h1>

<h2>2nd Place</h2>

<p>
	The qualifying round of the hackathon involved a Kaggle competition comprising two challenges:
</p>

<ul>
	<li>
		Text classification:
		<ul>
			<li>
				Given a natural sentence, perform multiclass classification to identify various key
				aspects, such as their gender as well as their clothes.
			</li>
			<li>
				Our team used a bi-directional LSTM with attention, as well as label smoothing, tuning it
				with five-fold cross-validation, and achieved the highest F1-score on the private
				leaderboard among all teams.
			</li>
		</ul>
	</li>
	<li>
		Object detection:
		<ul>
			<li>
				Given an image, draw bounding boxes indicating where in the image there are clothes, as
				well as what kind of clothes they are, such as trousers or dresses.
			</li>
			<li>Evaluation was based on mAP.</li>
			<li>
				Our team used an ensemble of 5 different object detection models, comprising YOLOv4,
				YOLOv5, EfficientDet, DETR, and RetinaNet, and combined their predictions using
				<a href="https://github.com/ZFTurbo/Weighted-Boxes-Fusion" target="_blank"
					>Weighted-Boxes-Fusion</a
				>. We used different image augmentation methods for each model to increase diversity.
			</li>
		</ul>
	</li>
</ul>

<p>Our team emerged as the top team on the final private leaderboard.</p>

<p>
	The second (and final) stage of the hackathon was a disaster rescue scenario involving several
	connected challenges. This stage was conducted physically, and all challenges below had to run
	consecutively without any human intervention.
</p>

<ul>
	<li>
		Aerial photography of disaster site using a drone:
		<ul>
			<li>
				We were tasked to write code that could fly a small drone upwards, locate a landmark on
				the ground, position itself over that landmark, and capture the scene.
			</li>
			<li>The captured scene was used as the input to the next part of the challenge.</li>
		</ul>
	</li>
	<li>
		Autonomous path finding and obstacle avoidance based on the aerial photograph:
		<ul>
			<li>
				The obstacles' colors were distinct from the ground's colors, so we used OpenCV to perform
				color filtering and segment the scene into unobstructed vs obstructed areas.
			</li>
			<li>
				The areas were subdivided into smaller grids that we ran the A* search algorithm on. The
				destination location was detected in the scene by using text detection software as it was
				marked out with a letter.
			</li>
			<li>
				The path found by the A* search algorithm was translated to real world measurements by
				scaling it based on an object with a known size in the scene. In our case, we used the
				letter that marked the destination as our reference.
			</li>
		</ul>
	</li>
	<li>
		Autonomous identification of target given a natural language input:
		<ul>
			<li>
				The "people" in the scene were all wearing distinct sets of clothing, for example one
				could be wearing a red shirt and jeans while another would be wearing a blue shirt and
				jeans. We got the robot to capture still images of all the "people" by having it rotate to
				roughly locate each "person", followed by using a person detection model (YOLOv4) to
				extract their bounding box. The image was then cropped to the bounding box (with some
				padding) and fed to the CV model.
			</li>
			<li>
				We then had to run the natural language input of the target given through our NLP model
				and compare its outputs with our CV model's output for each "person" to identify the
				correct target. Sample natural language input would be something like: "She wore grey long
				pants with a hoodie and had fair skin.".
			</li>
		</ul>
	</li>
	<li>
		Autonomous grabbing of target with a robotic arm:
		<ul>
			<li>
				We had to write code that could move the robot forward to the correct distance away from
				the target based on the camera feed and move the robotic arm to grab the target firmly.
			</li>
			<li>
				We rotate the robot roughly to the direction of the target, then use a person detection
				model to find the bounding box of the target, and made minor adjustments after each
				forward movement to ensure that the bounding box was still roughly in the middle, which
				would ensure that the robot was not straying off track.
			</li>
			<li>
				We used an empirical method to pre-determine some thresholds the bounding box would hit
				when the robot is close enough to grab the person, and made sure that these thresholds
				weren't overshot by scaling down the range of the robot's forward motion as it approached
				the target. Once the thresholds were hit, the grabbing action sequence would be activated,
				and it would grab the target accurately and firmly.
			</li>
		</ul>
	</li>
</ul>

<h1>March 2016: Singapore Science and Engineering Fair</h1>

<h2>Silver Award</h2>

<p>
	My project was about the classification of handwritten characters by applying feature extraction
	on images and subsequently feeding these features into a small and simple feedforward neural
	network for classification. The feature extraction involved the following steps:
</p>

<ul>
	<li>Binarization of the image via Otsu's method</li>
	<li>Skeletonization</li>
	<li>Removing all connected components that were too small based on a threshold</li>
	<li>Extracting a fixed sample of points by tracing the remaining skeletons in a fixed order</li>
</ul>

<p>
	This preprocessing helped my model achieve comparable results to the state of the art (at that
	point in time). I presented my methodology and results to 3 judges at the Engineering Fair in the
	form of a poster, and received the Silver Award for my efforts.
</p>
