# Decrypting-Dimensionality-Reduction

Today, we will see dimensionality reduction. We will cover several important feature extraction and feature selection techniques. Let’s begin the journey….

As the number of features increases, the model becomes more complex. The more the number of features, the more the chances of overfitting. A machine learning model that is trained on a large number of features, gets increasingly dependent on the data it was trained on and in turn overfitted, resulting in poor performance on real data, beating the purpose.

Let’s say you have a straight line 100 yards long and you dropped a penny somewhere on it. It wouldn’t be too hard to find. You walk along the line and it takes two minutes. Now let’s say you have a square 100 yards on each side and you dropped a penny somewhere on it. It would be pretty hard, like searching across two football fields stuck together. It could take days. Now a cube 100 yards across. That’s like searching a 30-story building the size of a football stadium. Ugh. The difficulty of searching through the space gets a lot harder as you have more dimensions.

Avoiding overfitting is a major motivation for performing dimensionality reduction. The fewer features our training data has, the lesser assumptions our model makes and the simpler it will be. But that is not all and dimensionality reduction has a lot more advantages to offer, like-
1. _Less misleading data means model accuracy improves._
2. _Less dimensions mean less computing. Less data means that algorithms train faster._
3. _Less data means less storage space required._
4. _Less dimensions allow usage of algorithms that are unfit for a large number of dimensions_
5. _Removes redundant features and noise._
6. _Reducing the dimensions of data to 2D or 3D may allow us to plot and visualize it precisely._


Dimensionality reduction could be done by both feature selection methods as well as feature engineering methods. Feature selection is the process of identifying and selecting relevant features for your sample. Feature engineering is manually generating new features from existing features, by applying some transformation or performing some operation on them.

We can remove features with low variance as they help a very little or no help in differentiating the data point with others. Because variance is dependent on scale, you should always normalize your features first.
We can also remove features with many missing values.

## Feature Extraction Techniques-

**Q. Why we can’t use a simple projection to project data into lower dimension?**

Ans. Because it doesn’t preserve global or local information about our data. While techniques like PCA preserve global information and t-SNE, LDA preserve local information too.

### PCA-

Principal component analysis (PCA) is an unsupervised algorithm that creates linear combinations of the original features. PCA is able to do this by maximizing variances and minimizing the reconstruction error by looking at pair wised distances. The new features are orthogonal, which means that they are uncorrelated. Furthermore, they are ranked in order of their “explained variance.” The first principal component (PC1) explains the most variance in your dataset, PC2 explains the second-most variance which is not explained by first principal component, and so on. You should always normalize your dataset before performing PCA because the transformation is dependent on scale. If you don’t, the features that are on the largest scale would dominate your new principal components.

Geometrically speaking, principal components represent the directions of the data that explain a maximal amount of variance, that is to say, the lines that capture most information of the data. The relationship between variance and information here, is that, the larger the variance carried by a line, the larger the dispersion of the data points along it, and the larger the dispersion along a line, the more the information it has.

Steps-

1. **STANDARDIZATION**- The aim of this step is to standardize the range of the continuous initial variables so that each one of them contributes equally to the analysis as we know PCA works on variance.
2. **COVARIANCE MATRIX COMPUTATION**- The aim of this step is to understand how the variables of the input data set are varying from the mean with respect to each other, or in other words, to see if there is any relationship between them.

_The covariance matrix is a p × p symmetric matrix (where p is the number of dimensions) that has as entries the covariances associated with all possible pairs of the initial variables._

3. **COMPUTE THE EIGENVECTORS AND EIGENVALUES OF THE COVARIANCE MATRIX TO IDENTIFY THE PRINCIPAL COMPONENTS**- What you firstly need to know about them is that they always come in pairs, so that every eigenvector has an eigenvalue. And their number is equal to the number of dimensions of the data. The eigenvectors of the Covariance matrix are actually the directions of the axes where there is the most variance(most information) and that we call Principal Components. And eigenvalues are simply the coefficients attached to eigenvectors, which give the amount of variance carried in each Principal Component. By ranking your eigenvectors in order of their eigenvalues, highest to lowest, you get the principal components in order of significance. After having the principal components, to compute the percentage of variance (information) accounted for by each component, we divide the eigenvalue of each component by the sum of eigenvalues
4. **FEATURE VECTOR**- The feature vector is simply a matrix that has as columns the eigenvectors of the components that we decide to keep. This makes it the first step towards dimensionality reduction, because if we choose to keep only p eigenvectors (components) out of n, the final data set will have only p dimensions.
5. **RECAST THE DATA ALONG THE PRINCIPAL COMPONENTS AXES**- In this step, which is the last one, the aim is to use the feature vector formed using the eigenvectors of the covariance matrix, to reorient the data from the original axes to the ones represented by the principal components (hence the name Principal Components Analysis). This can be done by multiplying the transpose of the original data set by the transpose of the feature vector.

         Final_Dataset= Feature_Vector.T * OriginalDataset.T (original dataset should be standardized)
    
An important thing to realize here is that, the principal components are less interpretable and don’t have any real meaning since they are constructed as linear combinations of the initial variables.

PCA is a versatile technique that works well in practice. It’s fast and simple to implement. PCA should be used mainly for variables which are strongly correlated. If the relationship is weak between variables, PCA does not work well to reduce data. Refer to the correlation matrix to determine. In general, if most of the correlation coefficients are smaller than 0.3, PCA will not help.

PCA is a rotation of data from one coordinate system to another. A common mistake new data scientists make is to apply PCA to non-continuous variables. While it is technically possible to use PCA on discrete variables, or categorical variables that have been one hot encoded variables, you should not. Categorical variables are not numerical at all, and thus have no variance structure.

On relatively small data sets, the first few components can explain almost all the variance in your data set. I’ve seen other data scientists mistakenly think that this means that the last few components can then be disregarded as trivial, and that first few components are the features that are most important. The only way PCA is a valid method of feature selection is if the most important variables are the ones that happen to have the most variation in them. However this is usually not true.

We can use it by- from sklearn.decomposition import PCA and then set n-components as number of features in output dataset.

Though PCA reduces dimensions but when dealing with multi-class data it’s necessary to reduce dimensions in a way that inter class separation is also taken care of. LDA is an algorithm used for the same.

Also, PCA is highly influenced by outliers.

### Linear Discriminant Analysis (LDA)-

LDA is like PCA means dimensionality reduction technique, but it focuses on maximizing the separability between known classes. It is used as a tool for classification, dimension reduction, and data visualization. It is the most commonly used dimensionality reduction technique in supervised learning.

Two criteria are used by LDA to create a new axis:

1. _Maximize the distance between means of the two classes._
2. _Minimize the variation within each class._

LDA approach is very similar to Principal Component Analysis, both are linear transformation techniques for dimensionality reduction, but also pursuing some differences;

• The earliest difference between LDA and PCA is that PCA can do more of features classification and LDA can do data classification.

• The shape and location of a real dataset change when transformed into another space under PCA, whereas, there is no change of shape and location on transformation to different spaces in LDA. LDA only provides more class separability.

• PCA can be expressed as an unsupervised algorithm since it avoids the class labels and focuses on finding directions( principal components) to maximize the variance in the dataset, in contrast to this, LDA is defined as supervised algorithms and computes the directions to present axes and to maximize the separation between multiple classes.

If you are willing to reduce the number of dimensions to 1, you can just project everything to the x-axis but LDA uses information from both the features to create a new axis which in turn minimizes the variance and maximizes the class distance of the two variables.

You can achieve this in three steps:

• Firstly, you need to calculate the separability between classes which is the distance between the mean of different classes. This is called the _between-class variance_.

![image](https://user-images.githubusercontent.com/65160713/131223587-3099109c-f245-4550-afea-e5a2a356978b.png)

• Secondly, calculate the distance between the mean and sample of each class. It is also called the _within-class variance_.

![image](https://user-images.githubusercontent.com/65160713/131223608-373cb530-4f5e-4fc6-81e4-6d30ffc94443.png)

• Finally, construct the lower-dimensional space which maximizes the between-class variance and minimizes the within-class variance. P is considered as the lower-dimensional space projection, also called Fisher’s criterion.

![image](https://user-images.githubusercontent.com/65160713/131223618-74798d2c-9fd2-419a-88e4-87febf119229.png)

Then we take dot product of our feature matrix X and P to obtain the new feature matrix.

The assumptions made by an LDA model about your data:

• Each variable in the data is shaped in the form of a bell curve when plotted, i.e. Gaussian.

• The values of each variable vary around the mean by the same amount on the average, i.e. each attribute has the same variance.

But Linear Discriminant Analysis fails when the mean of the distributions are shared, as it becomes impossible for LDA to find a new axis that makes both the classes linearly separable. In such cases, we use non-linear discriminant analysis.

There are many extensions and variations to the method. Some popular extensions include:

• **Quadratic Discriminant Analysis (QDA)**: Each class uses its own estimate of variance (or covariance when there are multiple input variables).

• **Flexible Discriminant Analysis (FDA)**: Where non-linear combinations of inputs is used such as splines.

• **Regularized Discriminant Analysis (RDA)**: Introduces regularization into the estimate of the variance (actually covariance), moderating the influence of different variables on LDA.

**_Remember, both PCA and LDA are linear techniques. They are effective only when our data is linearly separable._**

PCA performs better in case where number of samples per class is less. Whereas LDA works better with large dataset having multiple classes

We’ll see more of LDA later.

**Q. When to use LDA over PCA or vice versa?**

1. When dataset is small use PCA
2. If samples per class are not same then use PCA
3. For multiclass use LDA
4. If your dataset has normal distribution LDA is more preferable.

### Singular Value Decomposition(SVD)-

https://towardsdatascience.com/understanding-singular-value-decomposition-and-its-application-in-data-science-388a54be95d

In summary both uses nearly same approach. PCA uses eigen decomposition of the covariance matrix. A symmetric matrix transforms a vector by stretching or shrinking it along its eigenvectors. But this is not the case in non-symmetric matrix and hence we can’t use eigen decomposition for a non-symmetric matrix and here’s where SVD comes to play. We can also perform PCA by Singular value decomposition. Infact, sklearn’s PCA uses SVD instead of eigenvalue decomposition as there are many benefits of this and also this makes PCA able to handle sparse matrix too. So there’s generally no benefit for using SVD over PCA for dimensionality reduction. Other than dimensionality reduction, SVD have many applications.

### t-SNE(t-distributed Stochastic Neighborhood Embeddings)-

t-SNE is an unsupervised, non-linear technique primarily used for data exploration and visualizing high-dimensional data. The t-SNE algorithm calculates a similarity measure between pairs of instances in the high dimensional space and in the low dimensional space. It then tries to optimize these two similarity measures using a cost function.

t-SNE however is not a clustering approach since it does not preserve the inputs like PCA and the values may often change between runs so it’s purely for exploration. Of course, t-SNE is not the only method that uses local structure. There are a host of other methods but , t-SNE just works well in practice. It also solves the crowding problem(somewhat similar points in higher dimension collapsing on top of each other in lower dimensions) to it’s best as it can’t be solved completely. t-SNE tries to best preserve the distance between the data points in a same neighborhood during transformation and hence solving the crowding problem better than other techniques. Also it uses both local and global structure(but very less) which make it better than others. But preservance of global structure is not guaranteed.

For crowding problem refer to- https://www.youtube.com/watchv=hMUrZ708PFk&list=PLupD_xFct8mHqCkuaXmeXhe0ajNDu0mhZ&index=4

Because the distributions are distance based, all the data must be numeric. You should convert categorical variables to numeric ones by binary encoding or a similar method. It is also often useful to normalize the data, so each variable is on the same scale. This avoids variables with a larger numeric range dominating the analysis.

Steps-

#### Step 1: Creating a probability distribution

1. Suppose you pick a single point _xᵢ_ in the dataset.
2. Then, you define the probability of picking another point _xⱼ_ in the dataset as the neighbor as

    ![image](https://user-images.githubusercontent.com/65160713/131225222-6986b6bf-5639-49f8-b5bd-c0804931704b.png)

This probability is proportionate to the probability density of a Gaussian centered at _xᵢ_. For points that are far away, the probability of being picked as a neighbor deteriorates quickly, but never reaches 0. The Gaussian is a commonly used distribution and is a natural choice for a probabilistic measure of similarity between points.

In t-SNE, we want the number of neighbors to be roughly the same for all points to prevent any single point from wielding a disproportionate influence. This means that we want _σᵢ_ to be small for points in densely populated areas and large for sparse areas. Hence, we can’t keep standard deviation same for all points. The way we quantify this is by specifying a hyperparameter called the perplexity. The perplexity is basically the effective number of neighbors for any point, and t-SNE works relatively well for any value between 5 and 50. Larger perplexities will take more global structure into account, whereas smaller perplexities will make the embeddings more locally focused. Basically, the higher the perplexity is the higher value variance has.

### Step 2: Recreating the probability distribution

Let’s express the low-dimensional mapping of xᵢ as yᵢ. Our basic intuition is that we want to make the low-dimensional mappings express a similar probability distribution. We can do this by gaussian distribution too but it has a “short tail”, meaning nearby points are likely to be squashed together which causes crowding problem.
To spread the points out, we want a probability distribution that has a longer tail. This is why t-SNE uses a Student t-distribution with a single degree of freedom (which is also known as the Cauchy distribution). Concretely, the distribution is:

![image](https://user-images.githubusercontent.com/65160713/131225343-43aec17b-5112-4844-8cd8-c8624a15fb15.png)

Optimizing this distribution is done by conducting gradient descent on the KL-divergence between the distributions p and q. KL divergence is is a measure of how one probability distribution diverges from a second, expected probability distribution.

The gradient can be expressed as:

![image](https://user-images.githubusercontent.com/65160713/131225348-13513970-15fd-40f2-bf8c-57adce53f21c.png)

Intuitively, the gradient represents the strength and direction of attraction/repulsion between two points. A positive gradient represents an attraction, whereas a negative gradient represents a repulsion between the points. This “push-and-pull” eventually makes the points settle down in the low-dimensional space.

It also uses early exaggeration. It Controls how tight natural clusters in the original space are in the embedded space and how much space will be between them. For larger values, the space between natural clusters will be larger in the embedded space. Again, the choice of this parameter is not very critical.

When the embedded points are allowed to move around freely, there is a chance that unwanted clusters will form prematurely, causing the parameters to get stuck in local minima. To prevent this, t-SNE uses the “**early compression**” trick. This trick involves simply adding an L2-penalty to the cost function at the early stages of optimization to make the distance between embedded points remain small. The strength of this optimization is a hyperparameter, but t-SNE performs fairly robustly regardless of how you set it.

_However, after this process, the input features are no longer identifiable, and you cannot make any inference based only on the output of t-SNE. Also t-SNE doesn’t learn any mapping function from higher dimension to lower dimension. So we can’t convert a new data point at test time to lower dimension for prediction task. Hence, t-SNE can’t be used for dimensionality reduction but only for visualization and data exploration. On the other hand, I would not give the output of a t-SNE as input to a classifier. Mainly because t-SNE is highly non linear and somewhat random and you can get very different outputs depending with different runs and different values of perplexity._

### Limitations of t-SNE

1. Unlike methods like PCA, t-SNE is non-convex, meaning it has multiple local minima and is therefore much more difficult to optimize. This means, on every run it gives different result.
2. It’s One assumption is that the local structure of the manifold is linear. The reason this assumption is important is that the distance between neighboring points is measured in Euclidean distance, which assumes linearity
3. Preservance of global structure is not guaranteed. It is only preserved at large perplexity values. But we don’t use large values as then our visualization is of no use. Hence, we can’t preserve global structure to good extent.
4. tSNE consumes too much memory for its computations which becomes especially obvious when using large perplexity
5. It is highly recommended to use another dimensionality reduction method (e.g. PCA for dense data or TruncatedSVD for sparse data) to reduce the number of dimensions to a reasonable amount (e.g. 50) if the number of features is very high and then perform t-SNE for visualization. TruncatedSVD gives same result as SVD but is much faster than SVD.
6. t-SNE takes considerably longer time to execute on the same sample size of data than PCA. It can take several hours on million-sample datasets where PCA will finish in seconds or minutes. It has a quadratic time and space complexity in the number of data points. This makes it particularly slow and resource draining while applying it to data sets comprising of more than 10,000 observations.
7. PCA it is a mathematical technique, but t-SNE is a probabilistic one.
8. t-SNE method has no knowledge of the class labels; it is completely unsupervised. Still, it can find a representation of the data in 2-dimensions that clearly separates the classes, based on how close points are in the original space.

We can use it by-

    from sklearn.manifold import TSNE
    
![image](https://user-images.githubusercontent.com/65160713/131225428-a3834fc1-a311-45db-ae8f-3de99505fdbf.png)

So, t-SNE is better than other techniques for data visualization most of the times but not for dimensionality reduction. It is well suited for the visualization of high-dimensional datasets. It also has benefit in visualizing as it is a non-linear technique.

**We cannot interpret cluster sizes and inter-cluster distances and cluster density by t-SNE.**

#### Points to remember-

1. Run t-SNE multiple times for same parameters. As it can give different results every time, it is advisable to run t-SNE multiple times for better understanding.
2. Run t-SNE up to iteration till when shape stabilizes.
3. Try different perplexity values as each gives a different result and then interpret by considering all of them.

So, in real life, we run t-SNE many times and choose the best result.

Manifold learning is not another variation of PCA but a generalization. **Something that performs well in PCA is almost guaranteed to perform well in t-SNE or another manifold learning technique, since they are generalizations.** Much like an object that is an apple is also a fruit (a generalization), usually something is wrong if something does not yield a similar result as its generalization. On the other hand, if both methods fail, the data is probably inherently tricky to model.

### UMAP(Uniform Manifold Approximation and Projection)-
UMAP is a new technique by McInnes et al. that offers a number of advantages over t-SNE, most notably increased speed and better preservation of the data’s global structure. UMAP can project the 784-dimensional, 70,000-point MNIST dataset in less than 3 minutes, compared to 45 minutes for scikit-learn’s t-SNE implementation.

The biggest difference between the the output of UMAP when compared with t-SNE is this balance between local and global structure — UMAP is often better at preserving global structure in the final projection and speed. This means that the inter-cluster relations are potentially more meaningful than in t-SNE. However, it’s important to note that, because UMAP and t-SNE both necessarily warp the high-dimensional shape of the data when projecting to lower dimensions, any given axis or distance in lower dimensions still isn’t directly interpretable in the way of techniques such as PCA.

Note that, using t-SNE, it takes an extremely high perplexity (~1000) to begin to see the global structure emerge, and at such large perplexity values the time to compute is dramatically longer. It’s also notable that t-SNE projections vary widely from run to run, with different pieces of the higher-dimensional data projected to different locations. While UMAP is also a stochastic algorithm, it’s striking how similar the resulting projections are from run to run and with different parameters. This is due, again, to UMAP’s increased emphasis on global structure in comparison to t-SNE. UMAP is a general-purpose dimensionality reduction technique that can be used as preprocessing for machine learning which is not the case in t-SNE.

Here too, cluster size and distance between cluster doesn’t matter and it is always advisable to plot more than one graph.

Algorithms like t-SNE, UMAP, etc are also called as neighbor based algorithms.

#### How exactly it works-

It works much same like t-SNE but with just some differences which are-

1. UMAP uses **exponential probability distribution in high dimensions** but not necessarily Euclidean distances like tSNE but rather any distance can be plugged in. In addition, the probabilities are not normalized:

![image](https://user-images.githubusercontent.com/65160713/131225577-e93abdec-2d20-4f0a-851e-56664083b074.png)

_Here ρ is an important parameter that represents the distance from each i-th data point to its first nearest neighbor. This ensures the local connectivity of the manifold. In other words, this gives a locally adaptive exponential kernel for each data point, so the distance metric varies from point to point._

2. **UMAP does not apply normalization** to either high- or low-dimensional probabilities, which is very different from tSNE and feels weird. However, just from the functional form of the high- or low-dimensional probabilities one can see that they are already scaled for the segment [0, 1] and it turns out that the absence of normalization, like the denominator dramatically reduces time of computing the high-dimensional graph .
3. UMAP uses the number of nearest neighbors instead of perplexity.

![image](https://user-images.githubusercontent.com/65160713/131225606-007595a5-92c4-4925-ac93-9d80a2fe2402.png)

4. UMAP uses the family of curves 1 / (1+a*y^(2b)) for modelling distance probabilities in low dimensions, not exactly Student t-distribution but very-very similar, please note that again no normalization is applied:

![image](https://user-images.githubusercontent.com/65160713/131225629-a7f2a37d-d78f-4d0f-b9d5-fc275f58f67a.png)

_where a≈1.93 and b≈0.79 for default UMAP hyperparameters (in fact, for min_dist = 0.001). In practice, UMAP finds a and b from non-linear least-square fitting to the piecewise function with the min_dist hyperparameter:_

![image](https://user-images.githubusercontent.com/65160713/131225648-3381da0e-3054-44c0-bc43-5a87faeea250.png)

5. UMAP uses binary cross-entropy (CE) as a cost function instead of the KL-divergence like tSNE does.

![image](https://user-images.githubusercontent.com/65160713/131225664-d7035b80-2657-4d58-bceb-60563efffaf6.png)

_This additional (second) term in the CE cost function makes UMAP capable of capturing the global data structure in contrast to tSNE that can only model the local structure at moderate perplexity values._

6. Finally, UMAP uses the Stochastic Gradient Descent (SGD) instead of the regular Gradient Descent (GD) like tSNE / FItSNE, this both speeds up the computations and consumes less memory.

Refer to https://umap-learn.readthedocs.io/en/latest/how_umap_works.html for further details.

### Why UMAP Can Preserve Global Structure

In contrast to tSNE, UMAP uses Cross-Entropy (CE) as a cost function instead of the KL-divergence:

![image](https://user-images.githubusercontent.com/65160713/131225705-3ddade1b-2c6b-4303-9a43-71cb824d228c.png)

This leads to huge changes in the local-global structure preservation balance. At small values of X we get the same limit as for tSNE since the second term disappears because of the pre-factor and the fact that log-function is slower than polynomial function:

![image](https://user-images.githubusercontent.com/65160713/131225715-fec12935-c422-4c14-88cf-755ba406c660.png)

Therefore the Y coordinates are forced to be very small, i.e. Y → 0, in order to minimize the penalty. This is exactly like the tSNE behaves. However, in the opposite limit of large X, i.e. X→∞, the first term disappears, pre-factor of the second term becomes 1 and we obtain:

![image](https://user-images.githubusercontent.com/65160713/131225727-1f576a3d-cedb-40dc-9f40-64f5447b1dc0.png)

Here if Y is small, we get a high penalty because of the Y in the denominator of the logarithm, therefore Y is encouraged to be large so that the ratio under logarithm becomes 1 and we get zero penalty. Therefore we get Y → ∞ at X → ∞, so the global distances are preserved when moving from high- to low-dimensional space, exactly what we want.

### Why Exactly UMAP is Faster than tSNE-

1. First, we dropped the log-part in the definition of the number of nearest neighbors, i.e. not using the full entropy like tSNE:

![image](https://user-images.githubusercontent.com/65160713/131225756-5f577f46-3d78-4801-af56-0cf11956c552.png)

_Since algorithmically the log-function is computed through the Taylor series expansion, and practically putting a log-prefactor in front of the linear term does not add much since log-function is slower than the linear function, it is nice to skip this step entirely._

2. Second reason is that we omitted normalization of the high-dimensional and low-dimensional probability.
3. Stochastic Gradient Descent (SGD) was applied instead of the regular Gradient Descent (GD).
4. Increasing the number of dimensions in the original data set we introduce sparsity on the data, i.e. we get more and more fragmented manifold, i.e. sometimes there are dense regions, sometimes there are isolated points (locally broken manifold). UMAP solves this problem by introducing the local connectivity ρ parameter which glues together (to some extent) the sparse regions via introducing adaptive exponential kernel that takes into account the local data connectivity. This is exactly the reason why UMAP can (theoretically) work with any number of dimensions and does not need the pre-dimensionality reduction step (Autoencoder, PCA) before plugging it into the main dimensionality reduction procedure.

**_As long as tSNE uses KL-divergence as cost function, it can not compete against UMAP in global distance preservation._**

There are not known any disadvantages of umap till now.

For use you have to install the package-

    conda install -c conda-forge umap-learn

Refer to https://umap-learn.readthedocs.io/en/latest/api.html for it’s methods.

Refer to https://github.com/lmcinnes/umap for it’s use.

Refer to https://umap-learn.readthedocs.io/en/latest/parameters.html for all parameters use.

Refer to https://umap-learn.readthedocs.io/en/latest/embedding_space.html for knowing which embedding when to use.

_As we know that UMAP is also a stochastic algorithm , so it is always better to plot more than one graph to get better understanding of data._

We can also use multi-threading for very fast performance. Refer to https://umap-learn.readthedocs.io/en/latest/reproducibility.html for details.

Larger n-neighbors is used for supervised dimension reduction, than the unsupervised reduction. Also, providing class labels make it separate the classes much more clearly. If you have data with known classes and want to separate them while still having a meaningful embedding of individual points then supervised UMAP can provide exactly what you need. t-SNE can’t be used for supervised dimension reduction.

UMAP is much faster than t-SNE but still much slow. If your data have a lot of dimensions(300+), then it is worth trying first to reduce features to 40–50 by PCA and then further reducing by UMAP. Because applying UMAP on large data may lead to very much time and memory.

For smaller datasets, time and complexity doesn’t matter much as approximately they are same. These matter for larger datasets. The large the data gets, more the time and memory it takes. But, it is possible that some algorithm takes more time for smaller datasets in comparison to other algorithms, but takes less time for larger datasets in comparison to others and vice versa.

![image](https://user-images.githubusercontent.com/65160713/131225922-224ca432-d3d0-4edc-9f9b-2c494c2d80f6.png)

_Based on the slopes of the lines, for even larger datasets the difference between UMAP and t-SNE is only going to grow._

#### Easy summary of what we do in UMAP-

So suppose, we have a dataset of K dimension. If we capture the topological representation of the points in higher dimension, then we can transform them in lower dimension by keeping the same topology. So, for capturing the topological representation, we have to generate an open cover for it. So we generate cover like circles to our data points.

![image](https://user-images.githubusercontent.com/65160713/131225980-46e464c2-7636-4c98-96a2-46a7b66c6e1c.png)

We then join the data points under same cover with every other data point which is in touch with that cover, which result in this-

![image](https://user-images.githubusercontent.com/65160713/131225996-27bc42d4-8e04-476d-a981-373a18dbd51a.png)

But this leads to several disconnected components, so this can’t represent our topological representation.

So we set this as -a cover about a point stretches to the k-th nearest neighbor of the point, where k is the sample size we are using to approximate the local sense of distance. This means, Each point is given its own unique distance function, and we can simply select balls of radius one with respect to that local distance function!

![image](https://user-images.githubusercontent.com/65160713/131226018-9005f9ad-7ad6-434c-b8b7-55026f80913a.png)

A small choice of k means we want a very local interpretation which will more accurately capture fine detail structure and variation whereas larger k will carry more of global information.

We can further use fuzzy topology-where being in an open set in a cover is no longer a binary yes or no, but instead a fuzzy value between zero and one. Obviously the certainty that points are in a ball of a given radius will decay as we move away from the center of the ball.

![image](https://user-images.githubusercontent.com/65160713/131226025-4cb3abfd-6def-4561-b5ee-4a8a92f89f1f.png)

And remember for representing it as a topological representation, the whole graph should be connected means every point should be connected to atleast one other point i.e. we should have complete confidence that the fuzzy circle extends as far as the closest neighbor of each point.

We then join the data points under same cover with every other data point which is in touch with that cover.

![image](https://user-images.githubusercontent.com/65160713/131226027-f75df267-94f3-45c1-a8aa-8e42a5020246.png)

As we see, now it is connected graph and so we can capture it’s topological representation.

But, Each point has its own local metric associated to it, and from point a’s perspective the distance from point a to point b might be 1.5, but from the perspective of point b the distance from point b to point a might only be 0.6. Which point is right? How do we decide? Going back to our graph based intuition we can think of this as having directed edges with varying weights. Between any two points we might have up to two edges and the weights on those edges disagree with one another. And then to convert it to undirected graph, we apply the edge-weight combination formula across the whole graph.

We then transform it to lower dimension and then use cross entropy to keep the topological representation same. In lower dimension, we use euclidean matrix to measure the distance between points. We do same in t-SNE too(capturing distance in higher dimension and then projecting it to lower dimension and then optimizing data points to have same distance),but by using different approach. But t-SNE was not this appropriate and principled.

**So, UMAP is also same like t-SNE, but with just more appropriate and accurate approach.**

### Isomap-

Again a non-linear unsupervised dimensionality reduction manifold technique which uses the concept of isometric mapping and number of neighbors. Linear methods reduce the dimensions based on Euclidean distances whereas ISOMAP(Isometric mapping) uses **Geodesic distance** approach among the multivariate data points.

Isomap works on following steps:

1. It determines neighboring points based on manifold distance and connects the points within a fixed radius.
2. It calculates the Geodesic distance among the points that were determined in above step.
3. And then by eigenvalue decomposition on the geodesic distance metrics, it finds the low embedding of data.

In non-linear manifolds, the Euclidean metric for distance holds good if and only if neighborhood structure can be approximated as linear. If neighborhood contains holes, then Euclidean distances can be highly misleading. Hence, we use geodesic distance here.

![image](https://user-images.githubusercontent.com/65160713/131226119-6a4052b0-aa26-4a25-9700-5177065471ab.png)

Now, if we look at the 1-D mapping based on the Euclidean metric, we see that for points which are far apart(a & b) have been mapped poorly. Only the points which can be approximated to lie on a linear manifold(c & d) give satisfactory results. On the other hand, see the mapping with geodesic distances, it nicely approximates the close points as neighbors and far away points as distant. The geodesic distances between two points in the image are approximated by graph distance between the two points.

You can implement it by sklearn as-

    from sklearn.manifold import Isomap

But now this technique is not used much, as we have better techniques like tSNE, UMAP, so we will not go in detail further.

![image](https://user-images.githubusercontent.com/65160713/131226150-4c752387-cd3d-4bcc-887e-7ea1e0fe0935.png)

### Locally Linear Embedding(LLE)-

The LLE algorithm is an unsupervised non-linear method for dimensionality reduction. LLE first finds the k-nearest neighbors of the points. Then, it approximates each data vector as a weighted linear combination of its k-nearest neighbors. Finally, it computes the weights that best reconstruct the vectors from its neighbors, then produce the low-dimensional vectors best reconstructed by these weights. It optimizes the weights by using mean squared error.

_It can be thought of as a series of local Principal Component Analyses which are globally compared to find the best non-linear embedding._

One advantage of the LLE algorithm is that there is only one parameter to tune, which is the value of K, or the number of nearest neighbours to consider as part of a cluster.

It may sometimes perform better than PCA, and sometimes worse.

But this is also not used now, so we will not discuss this too.

### Independent Component Analysis-

ICA is a linear dimension reduction method, which transforms the dataset into columns of independent components. Blind Source Separation and the “cocktail party problem” are other names for it. It assumes that each sample of data is a mixture of independent components and it aims to find these independent components. Independent components means that there’s not a linear or non-linear dependency between 2 components.

The major difference between PCA and ICA is that PCA looks for uncorrelated factors while ICA looks for independent factors.

So, as PCA generated principal components, ICA generates independent components. But it can’t rank features as PCA. Hence, PCA gives much better result than ICA for dimension reduction and . But it is used in place of PCA when PCA’s assumptions are not fulfilled. Otherwise, it is rarely used for dimension reduction. But it’s used for different tasks. For example, ICA can be used to separate 2 signals such as 2 person’s voice in same audio, etc.

### Factor Analysis-

https://www.datacamp.com/community/tutorials/introduction-factor-analysis

The factor analysis describes the covariance relationships among many variables in terms of a few underlying, but unobservable, random quantities called factors. Factor analysis believes that the variables can be grouped by their correlations. It may be assumed that variables within a particular group are highly correlated among themselves, but they have relatively small correlations with variables in a different group. Then it can be said that each group of variables represents a single underlying construct (or factor) that is responsible for the observed correlations. Factor analysis can be viewed as an attempt to approximate the covariance matrix Σ.

1. Run factor analysis if you assume or wish to test a theoretical model of latent factors causing observed variables.
2. Run principal component analysis If you want to simply reduce your correlated observed variables to a smaller set of important independent composite variables.
3. PCA is useful for reducing the number of variables while retaining the most amount of information in the data, whereas EFA is useful for measuring unobserved (latent), error-free variables.
4. When variables don’t have anything in common, as in the example above, EFA won’t find a well-defined underlying factor, but PCA will find a well-defined principal component that explains the maximal amount of variance in the data.

You can use it by-

    from sklearn.decomposition import FactorAnalysis
    
### Multi-dimensional Scaling-

**Multi-Dimension Scaling** is a distance-preserving manifold learning method. All manifold learning algorithms assume the dataset lies on a smooth, non linear manifold of low dimension and that a mapping f: RD -> Rd (D>>d) can be found by preserving one or more properties of the higher dimension space.

MDS takes a dissimilarity matrix D(The Dissimilarity matrix is a matrix that expresses the similarity pair to pair between two sets.) where Dij represents the distance(generally euclidean) between points i and j and produces a mapping on a lower dimension, preserving the dissimilarities as closely as possible. Remember the dissimilarity/distance matrix is a symmetric matrix with diagonals equal to zero.

So, it preserves distance, means If 2 points are close in the feature space, it should be close in the latent factor space.

![image](https://user-images.githubusercontent.com/65160713/131226296-6f0afbd1-f9c5-4ee6-af75-6ed6cdd2444b.png)

It’s of two types-

#### Metric MDS-

We calculate the dissimilarity matrix of the data. Then we find an optimal configuration in lower dimension by optimizing a cost function.

![image](https://user-images.githubusercontent.com/65160713/131226341-e65bbbed-00fe-4b77-9bf2-819015e8c44b.png)

This loss function is also called as stress. So, given a distance or dissimilarity matrix D(X), MDS attempts to find D(Y) referred above as f (dᵢⱼ)with n data points y1, y2, ……. yn in p dimensions by optimizing the above loss function.

Metric MDS has a subtype known as Classical MDS which instead of optimizing cost function uses eigenvalue decomposition. It is more widely used so we will talk about it only. It is also called as Principal Coordinate Analysis(PCoA).

It is different than PCA as PCA creates plot based on correlations among samples while MDS creates plot based on distance between samples. The closer the samples are to each other, more close they are gonna be to each other.

![image](https://user-images.githubusercontent.com/65160713/131226379-5509902d-26e3-463b-9656-12d77d21f73a.png)

It calculates distance by calculating pairwise distance between each feature’s sample. We can calculate distance by any method like Euclidean distance. But on using Euclidean distance, it gives same result as PCA because clustering based on minimizing the linear distances is same as maximizing the linear correlations. Hence we generally use other measures. MDS is interpreted same as PCA means eigenvectors with larger eigenvalues explain most of the data.

**Metric MDS is suitable for quantitative data(which is not ordinal). For ordinal data, we use non-metric MDS.**

#### Non-metric MDS-

It’s also known as ordinal MDS. Here, it’s not the metric of a distance value that is important or meaningful, but its value in relation to the distances between other pairs of objects. It work on ranks-order instead of distance.

Metric multidimensional scaling creates a configuration of points whose inter-point distances approximate the given dissimilarities. This is sometimes too strict a requirement, and non-metric scaling is designed to relax it a bit. Instead of trying to approximate the dissimilarities themselves, non-metric scaling approximates a nonlinear, but monotonic, transformation of them. Because of the monotonicity, larger or smaller distances on a plot of the output will correspond to larger or smaller dissimilarities, respectively.

NMDS is a rank-based approach. This means that the original distance data is substituted with ranks. Thus, rather than object A being 2.1 units distant from object B and 4.4 units distant from object C, object C is the “first” most distant from object A while object C is the “second” most distant. While information about the magnitude of distances is lost, rank-based methods are generally more robust to data which do not have an identifiable distribution. Hence, it can also tolerate missing pairwise distance.

The basic steps in a non-metric MDS algorithm are:

1. Find a random configuration of points, e. g. by sampling from a normal distribution.
2. Calculate the distances d between the points.
3. Find the optimal monotonic transformation of the proximities, in order to obtain optimally scaled data f(x)(Monotonic transformation is a way of transforming a set of numbers into another set that preserves the order of the original set, it is a function mapping real numbers into real numbers, which satisfies the property, that if x>y, then f(x)>f(y), simply it is a strictly increasing function.)
4. Minimize the stress between the optimally scaled data and the distances by finding a new configuration of points.
5. Compare the stress to some criterion. If the stress is small enough then exit the algorithm else return to 2.

Let x denote the vector of random configuration, f(x) a monotonic transformation and d the point distances, then-

![image](https://user-images.githubusercontent.com/65160713/131226497-3624f876-d7fa-4a14-a048-4b2f3ad0b4aa.png)

MDS already has the input matrix in the form of distances (i.e. actual distances between cities) and therefore the distances have meaning in the input matrix and create a map of actual physical locations from those distances while in non-metric MDS, the distances are just a representation of the rankings (i.e. high as in 7 or low as in 1) and they do not have any meaning on their own but they are needed to create the map using euclidean geometry and the map then just shows the similarity in rankings represented by distances between coordinates on the map.

While PCA retains m important dimensions for you, Non-metric MDS fits configuration to m dimensions (you pre-define m) and it reproduces dissimilarities on the map more directly and accurately than PCA usually can

[_Often we care more about relative positioning than absolute differences, in which case non-metric is preferred to metric MDS._]

We can implement it by-

    from sklearn.manifold import MDS
    
#### PCA vs MDS-

MDS can be used on data for which we do not know coordinates, only the relative distances or if assumptions of PCA are not matched. Otherwise PCA is sufficient.

#### t-SNE vs MDS-

t-SNE is not designed to preserve distances while MDS does that. t-SNE just clusters same data but then it can place anywhere. For example if some data points are located on north side so t-SNE can cluster them together and place them on east, distorting the orientation. t-SNE is shown to preserve global structure better than classical multi-dimensional scaling which works on local pairs of data points only.

But MDS is suitable only for small datasets as for large datasets it takes much more time than other techniques. For small datasets it is faster than t-SNE but slower than other common dimension reduction techniques but for larger data it takes even more time than t-SNE.

![image](https://user-images.githubusercontent.com/65160713/131226527-3465c632-c5bf-4cb2-bce5-1a96dbce43af.png)

We also sometimes use sammon mapping with MDS rather than regular MDS as it can retain the local fine structure better.

_Isomap is also some-like MDS but it uses geodesic distance. Even t-SNE can be interpreted as MDS just with a special function._

## Feature Selection techniques-

### Filter methods-

Filter methods pick up the intrinsic properties of the features measured via univariate statistics instead of cross-validation performance. These methods are faster and less computationally expensive than wrapper methods. When dealing with high-dimensional data, it is computationally cheaper to use filter methods.

### Information Gain-

Information gain or IG measures how much information a feature gives about the class. Thus, we can determine which attribute in a given set of training feature is the most meaningful for discriminating between the classes to be learned.
So, we calculate information gain between target and other feature and select with highest scores. When it is used for feature selection, it is then called mutual information. It provides a measure of how much useful information is conveyed to the model by including a given feature. Therefore, including features with a high mutual information can be helpful for boosting model accuracy.

![image](https://user-images.githubusercontent.com/65160713/131226606-584069b2-263c-487d-a4ad-32fc11b6f45b.png)

It can be used as-

      from sklearn.feature_selection import mutual_info_classif for classification
      from sklearn.feature_selection import mutual_info_regression for regression

True mutual information can’t be negative. If its estimate turns out to be negative, it is replaced by zero.

It also retains non-linear relationship between features and target.

### Statistical test-

We use statistical tests like t-test, chi-squared, ANOVA to calculate p-value and then use it to do feature selection. 

So, we calculate significance of each feature in improving the model w.r.t target.

![image](https://user-images.githubusercontent.com/65160713/131226739-6aab9b85-f5ac-4f95-abdd-8e8d84835186.png)

These tests can only tell that WHICH FEATURES HAVE STRONGEST RELATIONSHIP WITH TARGET VARIABLE.

In the case of feature selection, the hypothesis we wish to test is along the lines of: True or False: This feature has no relevance to the response variable. We want to test this hypothesis for every feature and decide whether the features hold some significance in the prediction of the response. In a way, this is how we dealt with the correlation logic

We generally give these results of statistical tests or mutual information or other tests to sklearn’s [**SelectKBest**] or [**SelectPercentile**] , and these functions test above as test to provide output with desirable number of features.

One thing that should be kept in mind is that filter methods do not remove multicollinearity. So, you must deal with multicollinearity of features as well before training models for your data.

These were filter methods. Remember, they doesn’t depend on models but only on data unlike wrapper methods. They are very good for eliminating irrelevant, redundant, constant, duplicated, and correlated features.

### Correlation coefficient-

We generally remove the features with low correlation with target variables.

For linear relation we use Pearson’s correlation coefficient. If two variables are non-linear but monotonic, then we can use Spearman’s rank correlation coefficient. It’s a lot like Pearson’s correlation, but whereas Pearson’s correlation assesses linear relationships, Spearman’s correlation assesses monotonic relationships (whether linear or not).

Spearman’s coefficient is suitable for both continuous and discrete ordinal variables. If data is discrete but not ordinally then we use Kendall’s rank correlation coefficient.

We also remove the multicollinear features.

_The filter method looks at individual features for identifying it’s relative importance. A feature may not be useful on its own but maybe an important influencer when combined with other features. Filter methods may miss such features. But they are used as they are fast and don’t depend on model we use._

## Wrapper methods-

Wrappers require some method to search the space of all possible subsets of features, assessing their quality by learning and evaluating a classifier with that feature subset. The feature selection process is based on a specific machine learning algorithm that we are trying to fit on a given dataset. It follows a greedy search approach by evaluating all the possible combinations of features against the evaluation criterion. The wrapper methods usually result in better predictive accuracy than filter methods. They find the optimal feature subset for the desired machine learning algorithm Hence, they are very costly.

Wrapper methods work in the following way, generally speaking:

1. **Search for a subset of features:** Using a search method (described next), we select a subset of features from the available ones.
2. **Build a machine learning model:** In this step, a chosen ML algorithm is trained on the previously-selected subset of features.
3. **Evaluate model performance:** And finally, we evaluate the newly-trained ML model with a chosen metric.
4. **Repeat:** The whole process starts again with a new subset of features, a new ML model trained, and so on.
We stop until the desired condition is met, and then we choose the best subset with the best result in the evaluation phase.

We install library mlxtend to use wrapper methods.

### Forward Feature Selection-

Also known as step forward feature selection (or sequential forward feature selection — SFS), this is an iterative method in which we start by evaluating all features individually, and then select the one that results in the best performance. In the second step, it tests all possible combinations of the selected feature with the remaining features and retains the pair that produces the best algorithmic performance. And the loop continues by adding one feature at a time in each iteration until the pre-set criterion is reached.

We use it by-

    from mlxtend.feature_selection import SequentialFeatureSelector

And we set f_orward=True_ for forward feature selection.

### Backward feature Selection-

Same like forward feature selection, just do that in opposite way.

We start with all the features in the dataset, and then we evaluate the performance of the algorithm. After that, at each iteration, backward feature elimination removes one feature at a time, which produces the best performing algorithm using an evaluation metric. This feature can be also described as the least significant feature among the remaining available ones. And it continues, removing feature after feature until a certain criterion is satisfied.

We set _forward = False_ to use SequentialFeatureSelector as backward feature selector.

Apart from cost, above two have one more drawback, i.e. since we know that forward feature selector adds features at each iteration, a problem can occur when we add up a feature that was useful in the beginning, but after adding more ones, is now non-useful. At this point, there’s no way to remove this kind of feature. Same with backward selector but in reverse direction.

### Sequential Floating-

It is just an extension to solve the above problem.

The way this method works is quite simple to understand. Let’s explore it in the context of both methods:
1. **Step floating forward selection:** After each forward step, SFFS performs backward steps as long as the objective function increases.
2. **Step floating backward selection:** After each backward step, SFBS performs forward steps as long as the objective function increases.

To use it, we have to set floating parameter to be true of SequentialFeatureSelector.

### Recursive Feature Elimination-

This is just a fancy name for a simple method that works as follows:

1. Train a model on all the data features. This model can be a tree-based model, lasso, logistic regression, or others that can offer feature importance. Evaluate its performance on a suitable metric of your choice.
2. Derive the feature importance to rank features accordingly.
3. Delete the least important feature and re-train the model on the remaining ones.
4. Use the previous evaluation metric to calculate the performance of the resulting model.
5. Now test whether the evaluation metric decreases by an arbitrary threshold (you should define this as well). If it does, that means this feature is important. Otherwise, you can remove it.
6. Repeat steps 3–5 until all features are removed (i.e. evaluated).
You might be thinking to say that this is just like the step backward features selection that we did in our post on wrapper methods, but it isn’t. The difference is that SBS eliminates all the features first in order to determine which one is the least important. But here, we’re getting this information from the machine learning model’s derived importance, so it removes the feature only once rather than removing all the features at each step.

That’s why this approach is faster than pure wrapper methods and better than pure embedded methods. But as a drawback, the main problem with that is we have to use an arbitrary threshold value to decide whether to keep a feature or not.

As a consequence, the smaller this threshold value, the more features will be included in the subset, and vice versa.

It is considered as a hybrid method of embebbed method and wrapper method.

We can use it as-

    from sklearn.feature_importance import RFE

### Exhaustive Feature Selection-

Finally, this method searches across all possible feature combinations. Its aim is to find the best performing feature subset — we can say it’s a brute-force evaluation of feature subsets. It creates all the subsets of features from 1 to N, with N being the total number of features, and for each subset, it builds a machine learning algorithm and selects the subset with the best performance.

We use it by-

    from mlxtend.feature_selection import ExhaustiveFeatureSelector

As wrapper methods are very expensive, so first we remove features like which are multicollinear or some other criteria and then se wrapper methods to save some time.

Also note that using the subset of features from the wrapper methods make the model more prone to overfitting as compared to using subset of features from the filter methods.

## Embebbed methods-

Embedded methods complete the feature selection process within the construction of the machine learning algorithm itself. In other words, they perform feature selection during the model training, which is why we call them embedded methods. A learning algorithm takes advantage of its own variable selection process and performs feature selection and classification/regression at the same time.

The embedded method solves both issues we encountered with the filter and wrapper methods by combining their advantages. Here’s how:
1. They take into consideration the interaction of features like wrapper methods do.
2. They are faster like filter methods.
3. They are more accurate than filter methods.
4. They find the feature subset for the algorithm being trained.
5. They are much less prone to overfitting.

### Lasso Regression-

From the different types of regularization, Lasso or L1 has the property that is able to shrink some of the coefficients to zero. Therefore, that feature can be removed from the model.

### Random Forest Importance-

Random Forests is a kind of a Bagging Algorithm that aggregates a specified number of decision trees. The tree-based strategies used by random forests naturally rank by how well they improve the purity of the node, or in other words a decrease in the impurity (Gini impurity) over all trees. Nodes with the greatest decrease in impurity happen at the start of the trees, while notes with the least decrease in impurity occur at the end of trees. Thus, by pruning trees below a particular node, we can create a subset of the most important features.

But embebbed methods have disadvantage as they can be only used by certain algorithms.

## Other methods for feature selection-

### Permutation Importance-

It replaces a feature’s value with random noise and then train the model. A feature is “important” if shuffling its values increases the model error. In this case, the model relies on the feature for the prediction.

We can use it by-

    from eli5.sklearn import PermutationImportance

### Feature Importance-

You can get the feature importance of each feature of your dataset by using the feature importance property of the model. Feature importance gives you a score for each feature of your data, the higher the score more important or relevant is the feature towards your output variable. These can only work with model that have attribute __feature_importance__

### Autoencoders-
We will see this in my next blog.
