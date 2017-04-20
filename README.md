# CSC522 Course Project: GMM based Change Detection Framework Using Very High-resolution Satellite Imagery


High resolution images provided by satellites are good resource to use to identify and quantify landscapes changes. We explore a detection method to identify landscape changes using high resolution satellite images. This grid based method is helpful in Bi-temporal change detection. By given two satellite images from the same area, it can identify changes accurately. In our project, we divide each high-resolution satellite image into equally-sized square grids and fit each grid of RGB pixels using one Gaussian distribution. Next, we use symmetric Kullback-Leibler (KL) divergence of two Gaussian distributions at the same location of various time as metrics of image changes. Then a Gaussian Mixture Model (GMM) is used to cluster the KL divergence map into various levels of change, which is a good visualization of change detection between the two images over time.

## Getting Started

The models and algorithms of our project is implemented in Python code with the help of IPython Notebook for data and result visualization. All experiments are executed on local machine.

### Prerequisites

Following Python packages have to be installed before executing the project code

```
numpy
scipy
sklearn
skimage
matplotlib
```

### Installing

IPython notebook can be installed separately using pip 

```
pip install ipython
```

Or with Anaconda bundle.

```
conda update conda
conda update ipython
```

And it can be viewed using available web browser by the following command-line in terminal inside the directory of code:

```
ipython notebook
```

## Running the tests

Most of our code are recorded in ipython notebook cells. This notebook can be executed cell by cell in sequential order, or execute all at once using the Kernel starter. And the results will be visualized in images shown below the corresponding cells. We also extracted all codes and saved in python script which can be executed separately and save results in the very directory. To execute, use command-line in terminal:

```
python high_resolution_image_change_detection.py
```

### Expected Results

The final product of our code execution should be a pair of maps on the same landscape area with major changes marked in transparent color blocks. For example,

![Change Detection on Maps](https://github.com/jerry-shijieli/CSC522_GMM_on_Satellite_Image_Change_Detection/blob/master/result/change%20detection.png)

For more details and intermediate results, please check the ipython notebooks in the folder [code](https://github.com/jerry-shijieli/CSC522_GMM_on_Satellite_Image_Change_Detection/tree/master/code)

## Built With

* [Chrome](https://www.google.com/chrome/browser/desktop/) - The web brower used to view and edit ipython notebook
* [PyCharm](https://www.jetbrains.com/pycharm/) - Python IDE
* [Orfeo ToolBox (OTB)](https://www.orfeo-toolbox.org/CookBook/index_TOC.html) - Used to view and process satellite images

## Contributing

All team members contribute equivalently to this development of this project. See the final report for details.

## Versioning

We use both [git](https://git-scm.com) and hand-labeled file names for versioning. The existence of various versions at the same time is for the sake of result comparisons and code optimization.

## Authors

* **Shijie Li**  *(email: sli41@ncsu.edu)* 
* **Tianpei Xia**  *(email: txia4@ncsu.edu)* 
* **Jingjuan Deng**  *(email: jdeng8@ncsu.edu)* 
* **Zhiren Lu**  *(email: nzlu@ncsu.edu)* 
* **Zifan Nan**  *(email: znan@ncsu.edu)* 
* **Rui Liu**  *(email: rliu13@ncsu.edu)* 

## License

This project is licensed under the MIT License - see the [MIT LICENSE](https://choosealicense.com/licenses/mit/) file for details.

## Acknowledgments

* Thank Prof. Raju Vatsavai for the support and advice on this project.
* Thank all TAs of CSC522 course for the suggestion on both algorithms and software tools to faciliate project development.
* Thank all team members for contributions on both project idea design, software implementation and experiment documentation.

