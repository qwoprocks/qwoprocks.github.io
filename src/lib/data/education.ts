export interface Education {
	school: string;
	degree: string;
	period: string;
	note: string;
	gpa?: string;
	details: string[];
}

export const education: Education[] = [
	{
		school: 'Carnegie Mellon University',
		degree: "Master's in Machine Learning",
		period: '2024 – 2025',
		note: 'School of Computer Science, Pittsburgh, PA',
		gpa: '4.07 / 4.00',
		details: [
			'Research: <a href="https://jykoh.com/vwa" target="_blank">VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks</a>',
			'15-642 Machine Learning Systems',
			'10-617 Intermediate Deep Learning',
			'10-715 Advanced Introduction to Machine Learning',
			'10-718 Machine Learning in Practice',
			'16-726 Learning-Based Image Synthesis',
			'16-720 Computer Vision',
			'36-705 Intermediate Statistics',
			'10-725 Convex Optimization',
			'10-708 Probabilistic Graphical Models'
		]
	},
	{
		school: 'National University of Singapore',
		degree: 'BComp in Computer Science with Honours',
		period: '2019 – 2023',
		note: 'AI Focus Area (Distinction)',
		gpa: '4.88 / 5.00 (3.92 / 4.00)',
		details: [
			"Dean's List: AY2021/2022 Sem 1, AY2020/2021 Sem 1",
			'Top student: CS4243 Computer Vision (156 students), CS2106 Operating Systems (403 students), CS2103T Software Engineering (338 students)',
			'CS4243 Computer Vision and Pattern Recognition (A+)',
			'CS3241 Computer Graphics (A)',
			'CS4248 Natural Language Processing (A-)',
			'CS3243 Introduction to Artificial Intelligence (A-)',
			'CS2106 Introduction to Operating Systems (A+)',
			'CS3230 Design and Analysis of Algorithms (A)',
			'CS2040S Data Structures and Algorithms (A+)',
			'CS2103T Software Engineering (A+)',
			'CS2100 Computer Organisation (A+)',
			'CS2030 Programming Methodology II (A+)',
			'CS1231S Discrete Structures (A+)',
			'MA1101R Linear Algebra I (A+)'
		]
	},
	{
		school: 'The University of British Columbia',
		degree: 'Student Exchange Programme',
		period: '2022',
		note: 'Vancouver, Canada',
		gpa: '4.20 / 4.00',
		details: [
			'CPSC 340 Machine Learning and Data Mining (A+)',
			'STAT 305 Introduction to Statistical Inference (A+)',
			'Participated in the rock climbing club and the varsity outdoor club'
		]
	}
];
