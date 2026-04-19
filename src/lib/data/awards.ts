export interface Award {
	title: string;
	place: string;
	date: string;
	description: string;
	details?: string[];
}

export const awards: Award[] = [
	{
		title: 'Meta Global Hackathon',
		place: '1st',
		date: 'October 2022',
		description: 'Global first place among thousands of participants',
		details: [
			'Coding Challenge: LeetCode-style questions (medium to hard) with speed bonuses',
			'Linux Challenge: Find secrets inside the filesystem using Linux commands',
			'Find the Bug Challenge: Identify bugs in language-agnostic code snippets within time limits',
			'Quizzes: General knowledge about programming and Meta',
			'Product Thinking Challenge: 2-man team overall champion — proposed and presented a tech solution to a given problem statement before a panel of Meta judges'
		]
	},
	{
		title: 'Citi Hackoverflow',
		place: '1st',
		date: 'August 2021',
		description: "First place in Citi's annual hackathon"
	},
	{
		title: 'BrainHack Today I Learned',
		place: '2nd',
		date: 'July 2020',
		description: 'Second place in national-level hackathon combining ML and robotics',
		details: [
			'Qualifying round: Kaggle competition with text classification (bi-directional LSTM with attention, label smoothing, 5-fold CV — highest F1 on private leaderboard) and object detection (ensemble of YOLOv4, YOLOv5, EfficientDet, DETR, RetinaNet with <a href="https://github.com/ZFTurbo/Weighted-Boxes-Fusion" target="_blank" rel="noopener noreferrer">Weighted-Boxes-Fusion</a>)',
			'Final stage: Disaster rescue scenario with autonomous drone photography, A*-based path planning from aerial images, NLP-guided target identification using CV+NLP model matching, and robotic arm target acquisition'
		]
	},
	{
		title: 'Singapore Science and Engineering Fair',
		place: 'Silver',
		date: 'March 2016',
		description: 'Handwritten character classification using feature extraction and neural networks',
		details: [
			'Feature extraction pipeline: Otsu binarization, skeletonization, connected component filtering, fixed-point skeleton tracing',
			'Achieved comparable results to state of the art at the time with a simple feedforward neural network'
		]
	}
];
