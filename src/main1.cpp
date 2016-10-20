#include <iostream>
#include <vector>
#include <string>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pcl/console/parse.h>
#include <pcl/console/time.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common_headers.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/visualization/cloud_viewer.h>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>

#include <stdio.h>
#include <math.h>
#include <boost/foreach.hpp>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/Dense>

typedef pcl::PointXYZI PointT;

using std::vector;
using std::string;
using namespace cv;
using std::cout;
using std::cerr;
using std::endl;
static bool verbose = false;

using namespace std;
using namespace Eigen;

/********************************************************************************************************
kフレーム、k+1フレーム、k+2フレームの点群データを入力．
各フレームの点群に対し、地面除去，クラスタリング，人物候補の抽出をした後、k+2フレームの人物候補の大きさを基準に
各フレームの人物候補から対応する候補を見つけ、重畳する．

<地面除去>
範囲はLIDARを中心に左右に20[m]ずつ、前に70[m]、後ろに20[m]としている．この範囲で2次元グリッドマップを生成する．
グリッドマップを生成する際の格子の大きさはLIDAR前方30[m]地点をさかいに変えた．30[m]以下は0.5[m]、以上は1[m]．
地面かどうかの判定方法もLIDAR前方30[m]地点をさかいに変えた．30[m]以下は分散値をもとに、以上は高さ情報をもとに
地面か否かの判定を行っている．
<クラスタリング>
探索距離の設定をLIDAR前方30[m]地点をさかいに変えた．30[m]以下は0.25[m]、以上は0.6[m]．
<人物候補抽出>
高さの条件：0.45~2.0[m]
幅の条件：0.05[m]~1.2[m]
奥行きの条件：~1.2[m]
<識別>
クラスタから特徴量を算出し、人物か否か判定する．
<同一人物かの判定・重畳>
クラスタリングの際、LIDAR前方30[m]地点以上の人物候補は高さ、幅、クラスタ内点群の最小最大座標点をベクタに格納．
これらを用いて人物候補のフレーム間での対応を求める．

地面除去：197~270行、272~338行、340~406行
クラスタリング（候補抽出・識別含む）：408~648行、650~884行、886~1120行
重畳処理：1122~1261行

********************************************************************************************************/

int main (int argc, char* argv[])
{
  // kフレーム、k+1フレーム、k+2フレームの点群を格納するポインタ
  pcl::PointCloud<PointT>::Ptr k_cloud (new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr k1_cloud (new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr k2_cloud (new pcl::PointCloud<PointT>);
  // 地面除去した点群を格納するポインタ
  pcl::PointCloud<PointT>::Ptr filter_cloud (new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr filter_cloud1 (new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr filter_cloud2 (new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr filter_cloud3 (new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr filter_cloud4 (new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr filter_cloud5 (new pcl::PointCloud<PointT>);
  // viewport
  int V1 (0);
  int V2 (0);
  int V3 (0);
  int V4 (0);
  // ID
  char Box1[100];
  char Box2[100];
  char Box3[100];
  char name1[100];
  char name2[100];
  char name3[100];
  char supercloud[100];
  char superbox[100];
  char loadfile1[100];
  char loadfile2[100];
  char loadfile3[100];
  // クラスタリングに使うインデックス
  std::vector<pcl::PointIndices> cluster_indices;
  std::vector<pcl::PointIndices> cluster_indices1;
  std::vector<pcl::PointIndices> cluster_indices2;
  std::vector<pcl::PointIndices> cluster_indices3;
  std::vector<pcl::PointIndices> cluster_indices4;
  std::vector<pcl::PointIndices> cluster_indices5;
  int i = 0;
  int i1 = 0;
  int i2 = 0;
  int i3 = 0;
  int i4 = 0;
  int i5 = 0;	
  // 特徴量
  int n;
  float f11,f12,f13;
  float f21,f22,f23,f24,f25,f26;
  float v1, v2, v3;
  double v11, v22, v33;
  float f31,f32,f33,f34,f35,f36,f37;
  //ベクタ
  std::vector<PointT> minp_h_k;
  std::vector<PointT> maxp_h_k;
  std::vector<PointT> minp_h_k1;
  std::vector<PointT> maxp_h_k1;
  std::vector<PointT> minp_h_k2;
  std::vector<PointT> maxp_h_k2;
  std::vector<float> y_k;
  std::vector<float> z_k;
  std::vector<float> y_k1;
  std::vector<float> z_k1;
  std::vector<float> y_k2;
  std::vector<float> z_k2;
  std::vector<pcl::PointCloud<PointT>::Ptr> human_k;
  std::vector<pcl::PointCloud<PointT>::Ptr> human_k1;
  std::vector<pcl::PointCloud<PointT>::Ptr> human_k2;
  //座標変換行列
  float theta = 0;
  Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();
  transform_1 (0,0) = cos (theta);
  transform_1 (0,1) = -sin (theta);
  transform_1 (1,0) = sin (theta);
  transform_1 (1,1) = cos (theta);
  Eigen::Matrix4f transform_2 = Eigen::Matrix4f::Identity();
  transform_2 (0,0) = cos (theta);
  transform_2 (0,1) = -sin (theta);
  transform_2 (1,0) = sin (theta);
  transform_2 (1,1) = cos (theta);	
  // 設定
  pcl::visualization::PointCloudColorHandlerGenericField<PointT> handler(k_cloud, "intensity");
  pcl::PassThrough<PointT> pass;
  pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
  pcl::EuclideanClusterExtraction<PointT> ec;
  pcl::visualization::PCLVisualizer viewer ("visualizer");
  pcl::MomentOfInertiaEstimation <PointT> feature_extractor;
  pcl::console::TicToc time;

  // ファイル読み込み
  int file1 = atoi(argv[1]);
  int file2 = atoi(argv[2]);
  int file3 = atoi(argv[3]);
  sprintf (loadfile1, "master/master_%i.pcd", file1);
  sprintf (loadfile2, "master/master_%i.pcd", file2);
  sprintf (loadfile3, "master/master_%i.pcd", file3);
  pcl::io::loadPCDFile<PointT> (loadfile1, *k_cloud);
  pcl::io::loadPCDFile<PointT> (loadfile2, *k1_cloud);
  pcl::io::loadPCDFile<PointT> (loadfile3, *k2_cloud);

  // xmlファイル読み込み
  CvSVM svm;
  svm.load ("svm_2.xml");

  // create_viewport
  viewer.createViewPort(0.0, 0.5, 0.5, 1.0, V1);
  viewer.createViewPort(0.5, 0.5, 1.0, 1.0, V2);
  viewer.createViewPort(0.0, 0.0, 0.5, 0.5, V3);
  viewer.createViewPort(0.5, 0.0, 1.0, 0.5, V4);
  /*
	viewer.createViewPort(0.0, 0.0, 1.0, 1.0, V1);
  */
  
  // cloudをvisualize
  viewer.addPointCloud<PointT> (k_cloud, handler,"k_cloud", V1);
  viewer.addPointCloud<PointT> (k1_cloud, handler,"k1_cloud", V2);
  viewer.addPointCloud<PointT> (k2_cloud, handler,"k2_cloud1", V3);
  viewer.addPointCloud<PointT> (k2_cloud, handler,"k2_cloud2", V4);
   	
  // 平面除去(k_cloud)
  time.tic ();
  // 30m以内
  // x方向の範囲を絞る
  for (float mx = -20.0; mx < 20.0; mx = mx + 0.5){
	pcl::PointCloud<PointT>::Ptr cloud_x (new pcl::PointCloud<PointT>);
	pass.setInputCloud (k_cloud);
	pass.setFilterFieldName ("x");
	pass.setFilterLimits (mx, mx+0.5);
	pass.filter (*cloud_x);
	// y方向の範囲を絞る
	for (float my = -20.0; my < 30.0; my = my + 0.5){
	  pcl::PointCloud<PointT>::Ptr cloud_y (new pcl::PointCloud<PointT>);
	  pass.setInputCloud (cloud_x);
	  pass.setFilterFieldName ("y");
	  pass.setFilterLimits (my, my+0.5);
	  pass.filter (*cloud_y);
	  if (cloud_y->points.size () >= 10){// 格子内の点が10以上だったら以下の操作を行う．
		// 分散値を基に地面除去
		int pointsize;
		float all_z = 0.0;// zの値をすべて足した値
		float ave_z;// zの値の平均
		float all_zz = 0.0;// 各zの値と平均値との差の2乗をすべて足した値
		float cov_z;// zの分散値
		for (size_t k = 0; k < cloud_y->points.size (); k++){
		  all_z = all_z + cloud_y->points[k].z;
		}
		pointsize = cloud_y->points.size ();
		ave_z = all_z / pointsize;
		for (int kk = 0; kk < cloud_y->points.size (); kk++){
		  all_zz = all_zz + pow(cloud_y->points[kk].z - ave_z, 2.0);
		}
		cov_z = all_zz / pointsize;
		if (cov_z  > 0.05){// 閾値0.05[m]
		  *filter_cloud = *filter_cloud + *cloud_y;// filter_cloudに地面が除去された点群が格納されていく．
		}	  
	  }
	}
  }
  // 30m以上
  // x方向の範囲を絞る
  for (float mx = -20.0; mx < 20.0; mx = mx + 1.0){
	pcl::PointCloud<PointT>::Ptr cloud_x1 (new pcl::PointCloud<PointT>);
	pass.setInputCloud (k_cloud);
	pass.setFilterFieldName ("x");
	pass.setFilterLimits (mx, mx+1.0);
	pass.filter (*cloud_x1);
	// y方向の範囲を絞る
	for (float my = 30.0; my < 70.0; my = my + 1.0){
	  pcl::PointCloud<PointT>::Ptr cloud_y1 (new pcl::PointCloud<PointT>);
	  pass.setInputCloud (cloud_x1);
	  pass.setFilterFieldName ("y");
	  pass.setFilterLimits (my, my+1.0);
	  pass.filter (*cloud_y1);

	  if (cloud_y1->points.size () >= 1){// 格子内の点が1以上だったら以下の操作を行う．
		if (cloud_y1->points.size () <= 3){
		  *filter_cloud1 = *filter_cloud1 + *cloud_y1;// 格子内の点が3以下だったら、filter_cloud1に格納する．
		} else {
		  // 高さ情報を基に地面除去
		  float dif = 0.0;
		  PointT minp, maxp;
		  feature_extractor.setInputCloud (cloud_y1);
		  feature_extractor.compute ();
		  feature_extractor.getAABB (minp, maxp);
		  dif = maxp.z - minp.z;
		  if (std::abs(dif) > 0.15){// 閾値0.15[m]
			*filter_cloud1 = *filter_cloud1 + *cloud_y1;// filter_cloud1に地面が除去された点群が格納されていく．
		  }
		}
	  }
	}
  }
  cout << "平面除去1の処理時間：" << time.toc () << "[ms]" << endl;

  // 平面除去(k1_cloud)
  time.tic ();
  // 30m以内
  for (float mx = -20.0; mx < 20.0; mx = mx + 0.5){
	pcl::PointCloud<PointT>::Ptr cloud_x2 (new pcl::PointCloud<PointT>);
	pass.setInputCloud (k1_cloud);
	pass.setFilterFieldName ("x");
	pass.setFilterLimits (mx, mx+0.5);
	pass.filter (*cloud_x2);
	for (float my = -20.0; my < 30.0; my = my + 0.5){
	  pcl::PointCloud<PointT>::Ptr cloud_y2 (new pcl::PointCloud<PointT>);
	  pass.setInputCloud (cloud_x2);
	  pass.setFilterFieldName ("y");
	  pass.setFilterLimits (my, my+0.5);
	  pass.filter (*cloud_y2);
	  if (cloud_y2->points.size () >= 10){
		int pointsize;
		float all_z = 0.0;
		float ave_z;
		float all_zz = 0.0;
		float cov_z;
		for (size_t k = 0; k < cloud_y2->points.size (); k++){
		  all_z = all_z + cloud_y2->points[k].z;
		}
		pointsize = cloud_y2->points.size ();
		ave_z = all_z / pointsize;
		for (int kk = 0; kk < cloud_y2->points.size (); kk++){
		  all_zz = all_zz + pow(cloud_y2->points[kk].z - ave_z, 2.0);
		}
		cov_z = all_zz / pointsize;
		if (cov_z  > 0.05){
		  *filter_cloud2 = *filter_cloud2 + *cloud_y2;
		}	  
	  }
	}
  }
  // 30m以上
  for (float mx = -20.0; mx < 20.0; mx = mx + 1.0){
	pcl::PointCloud<PointT>::Ptr cloud_x3 (new pcl::PointCloud<PointT>);
	pass.setInputCloud (k1_cloud);
	pass.setFilterFieldName ("x");
	pass.setFilterLimits (mx, mx+1.0);
	pass.filter (*cloud_x3);
	for (float my = 30.0; my < 70.0; my = my + 1.0){
	  pcl::PointCloud<PointT>::Ptr cloud_y3 (new pcl::PointCloud<PointT>);
	  pass.setInputCloud (cloud_x3);
	  pass.setFilterFieldName ("y");
	  pass.setFilterLimits (my, my+1.0);
	  pass.filter (*cloud_y3);
	  if (cloud_y3->points.size () >= 1){
		if (cloud_y3->points.size () <= 3){
		  *filter_cloud3 = *filter_cloud3 + *cloud_y3;
		} else {
		  float dif = 0.0;
		  PointT minp, maxp;
		  feature_extractor.setInputCloud (cloud_y3);
		  feature_extractor.compute ();
		  feature_extractor.getAABB (minp, maxp);
		  dif = maxp.z - minp.z;
		  if (std::abs(dif) > 0.15){
			*filter_cloud3 = *filter_cloud3 + *cloud_y3;
		  }
		}
	  }
	}
  }
  cout << "平面除去2の処理時間：" << time.toc () << "[ms]" << endl;

  // 平面除去(k2_cloud)
  time.tic ();
  // 30m以内
  for (float mx = -20.0; mx < 20.0; mx = mx + 0.5){
	pcl::PointCloud<PointT>::Ptr cloud_x4 (new pcl::PointCloud<PointT>);
	pass.setInputCloud (k2_cloud);
	pass.setFilterFieldName ("x");
	pass.setFilterLimits (mx, mx+0.5);
	pass.filter (*cloud_x4);
	for (float my = -20.0; my < 30.0; my = my + 0.5){
	  pcl::PointCloud<PointT>::Ptr cloud_y4 (new pcl::PointCloud<PointT>);
	  pass.setInputCloud (cloud_x4);
	  pass.setFilterFieldName ("y");
	  pass.setFilterLimits (my, my+0.5);
	  pass.filter (*cloud_y4);
	  if (cloud_y4->points.size () >= 10){
		int pointsize;
		float all_z = 0.0;
		float ave_z;
		float all_zz = 0.0;
		float cov_z;
		for (size_t k = 0; k < cloud_y4->points.size (); k++){
		  all_z = all_z + cloud_y4->points[k].z;
		}
		pointsize = cloud_y4->points.size ();
		ave_z = all_z / pointsize;
		for (int kk = 0; kk < cloud_y4->points.size (); kk++){
		  all_zz = all_zz + pow(cloud_y4->points[kk].z - ave_z, 2.0);
		}
		cov_z = all_zz / pointsize;
		if (cov_z  > 0.05){
		  *filter_cloud4 = *filter_cloud4 + *cloud_y4;
		}	  
	  }
	}
  }
  // 30m以上	
  for (float mx = -20.0; mx < 20.0; mx = mx + 1.0){
	pcl::PointCloud<PointT>::Ptr cloud_x5 (new pcl::PointCloud<PointT>);
	pass.setInputCloud (k2_cloud);
	pass.setFilterFieldName ("x");
	pass.setFilterLimits (mx, mx+1.0);
	pass.filter (*cloud_x5);
	for (float my = 30.0; my < 70.0; my = my + 1.0){
	  pcl::PointCloud<PointT>::Ptr cloud_y5 (new pcl::PointCloud<PointT>);
	  pass.setInputCloud (cloud_x5);
	  pass.setFilterFieldName ("y");
	  pass.setFilterLimits (my, my+1.0);
	  pass.filter (*cloud_y5);
	  if (cloud_y5->points.size () >= 1){
		if (cloud_y5->points.size () <= 3){
		  *filter_cloud5 = *filter_cloud5 + *cloud_y5;
		} else {
		  float dif = 0.0;
		  PointT minp, maxp;
		  feature_extractor.setInputCloud (cloud_y5);
		  feature_extractor.compute ();
		  feature_extractor.getAABB (minp, maxp);
		  dif = maxp.z - minp.z;
		  if (std::abs(dif) > 0.15){
			*filter_cloud5 = *filter_cloud5 + *cloud_y5;
		  }
		}
	  }
	}
  }
  cout << "平面除去3の処理時間：" << time.toc () << "[ms]" << endl;
	
  // クラスタリング(k_cloud)
  // 30m以内
  time.tic ();
  tree->setInputCloud (filter_cloud);
  ec.setClusterTolerance (0.25);
  ec.setMinClusterSize (10);
  ec.setMaxClusterSize (100000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (filter_cloud);
  ec.extract (cluster_indices);

  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it) {
	PointT minp1, maxp1;
	pcl::PointCloud<PointT>::Ptr cloud_cluster (new pcl::PointCloud<PointT>);

	for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
	  cloud_cluster->points.push_back (filter_cloud->points[*pit]);		
	cloud_cluster->width = cloud_cluster->points.size ();
	cloud_cluster->height = 1;
	cloud_cluster->is_dense = true;
	// クラスタの大きさを求める．
	feature_extractor.setInputCloud (cloud_cluster);
	feature_extractor.compute ();
	feature_extractor.getAABB (minp1, maxp1);
	Eigen::Vector3f dx(maxp1.x-minp1.x, 0.0, 0.0);
	Eigen::Vector3f dy(0.0, maxp1.y-minp1.y, 0.0);
	Eigen::Vector3f dz(0.0, 0.0, maxp1.z-minp1.z);
	f13 = dx.norm();// 幅
	f12 = dy.norm();// 奥行き
	f11 = dz.norm();// 高さ

	if ((f11 >= 0.45)&&(f11 <= 2.0)&&(f12 <= 1.2)&&(f13 >= 0.05)&&(f13 <= 1.2)){// 人物候補の判定
	  i++;

	  cv::Mat sample(1, 13, CV_32FC1);

	  // 共分散行列と中心を求める
	  EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;
	  Eigen::Matrix3f cmat;
	  Eigen::Vector4f xyz_centroid;
	  pcl::compute3DCentroid (*cloud_cluster, xyz_centroid);
	  pcl::computeCovarianceMatrix (*cloud_cluster, xyz_centroid, covariance_matrix);

	  n = cloud_cluster->points.size();
	  f21 = covariance_matrix (0,0) / (n - 1);
	  f22 = covariance_matrix (0,1) / (n - 1);
	  f23 = covariance_matrix (0,2) / (n - 1);
	  f24 = covariance_matrix (1,1) / (n - 1);
	  f25 = covariance_matrix (1,2) / (n - 1);
	  f26 = covariance_matrix (2,2) / (n - 1);
	  cmat (0,0) = f21;
	  cmat (0,1) = f22;
	  cmat (0,2) = f23;
	  cmat (1,0) = f22;
	  cmat (1,1) = f24;
	  cmat (1,2) = f25;
	  cmat (2,0) = f23;
	  cmat (2,1) = f25;
	  cmat (2,2) = f26;

	  // 固有値を求める
	  EigenSolver<MatrixXf> es(cmat, false);
	  complex<double> lambda1 = es.eigenvalues()[0];
	  complex<double> lambda2 = es.eigenvalues()[1];
	  complex<double> lambda3 = es.eigenvalues()[2];
	  v11 = lambda1.real();
	  v22 = lambda2.real();
	  v33 = lambda3.real();
	  std::vector<double> v;
	  v.push_back (v11);
	  v.push_back (v22);
	  v.push_back (v33);
	  std::sort(v.begin(), v.end() );
	  v3 = v[0];
	  v2 = v[1];
	  v1 = v[2];
	  v.clear();

	  double nnn = v1 * v2 * v3;
	  f31 = (v1 - v2) / v1;
	  f32 = (v2 - v3) / v1;
	  f33 = v3 / v1;
	  f34 = pow(nnn, 1.0 / 3.0);
	  f35 = (v1 - v3) / v1;
	  f36 = -(v1 * log(v1) + v2 * log(v2) + v3 * log(v3));
	  f37 = v3 / (v1 + v2 + v3);

	  // 特徴ベクトルの生成
	  std::vector<float> features;
	  features.push_back(f21);
	  features.push_back(f22);
	  features.push_back(f23);
	  features.push_back(f24);
	  features.push_back(f25);
	  features.push_back(f26);
	  features.push_back(f31);
	  features.push_back(f32);
	  features.push_back(f33);
	  features.push_back(f34);
	  features.push_back(f35);
	  features.push_back(f36);
	  features.push_back(f37);
	  
	  // 識別
	  for(int j=0; j<13; j++) {
		sample.at<float>(0,j) = features[j];
	  }
	  float res = svm.predict(sample);
	
	  sprintf (Box1, "Box1.%i", i);
	  if (res == 1) {// 人物
		viewer.addCube (minp1.x, maxp1.x, minp1.y, maxp1.y, minp1.z, maxp1.z, 0.0, 1.0, 0.0, Box1, V1);// greenbox
	  } else {// 非人物
		viewer.addCube (minp1.x, maxp1.x, minp1.y, maxp1.y, minp1.z, maxp1.z, 1.0, 0.0, 0.0, Box1, V1);// redbox
	  }
	}
  }
  cout << "クラスタリング1の処理時間：" << time.toc () << "[ms]" << endl;

  // 30m以上
  time.tic ();
  tree->setInputCloud (filter_cloud1);
  ec.setClusterTolerance (0.6);
  ec.setMinClusterSize (3);
  ec.setMaxClusterSize (100000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (filter_cloud1);
  ec.extract (cluster_indices1);
  
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices1.begin (); it != cluster_indices1.end (); ++it) {
	PointT minp1, maxp1;
	pcl::PointCloud<PointT>::Ptr cloud_cluster (new pcl::PointCloud<PointT>);

	for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
	  cloud_cluster->points.push_back (filter_cloud1->points[*pit]);		
	cloud_cluster->width = cloud_cluster->points.size ();
	cloud_cluster->height = 1;
	cloud_cluster->is_dense = true;

	feature_extractor.setInputCloud (cloud_cluster);
	feature_extractor.compute ();
	feature_extractor.getAABB (minp1, maxp1);
	Eigen::Vector3f dx(maxp1.x-minp1.x, 0.0, 0.0);
	Eigen::Vector3f dy(0.0, maxp1.y-minp1.y, 0.0);
	Eigen::Vector3f dz(0.0, 0.0, maxp1.z-minp1.z);
	f13 = dx.norm();
	f12 = dy.norm();
	f11 = dz.norm();

	if ((f11 >= 0.45)&&(f11 <= 2.0)&&(f12 <= 1.2)&&(f13 >= 0.05)&&(f13 <= 1.2)){
	  i++;
	  sprintf (name1, "1.%i", i2);
	  viewer.addText3D (name1, cloud_cluster->points[0], 0.25, 1.0, 1.0);
	  i2++;

	  // 幅、高さ、最小座標点、最大座標点、点群をベクタに格納
	  y_k.push_back(f13);
	  z_k.push_back(f11);
	  minp_h_k.push_back(minp1);
	  maxp_h_k.push_back(maxp1);
	  human_k.push_back(cloud_cluster);
		    
	  cv::Mat sample(1, 13, CV_32FC1);

	  // 共分散行列と中心を求める
	  EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;
	  Eigen::Matrix3f cmat;
	  Eigen::Vector4f xyz_centroid;
	  pcl::compute3DCentroid (*cloud_cluster, xyz_centroid);
	  pcl::computeCovarianceMatrix (*cloud_cluster, xyz_centroid, covariance_matrix);
	  n = cloud_cluster->points.size();
	  f21 = covariance_matrix (0,0) / (n - 1);
	  f22 = covariance_matrix (0,1) / (n - 1);
	  f23 = covariance_matrix (0,2) / (n - 1);
	  f24 = covariance_matrix (1,1) / (n - 1);
	  f25 = covariance_matrix (1,2) / (n - 1);
	  f26 = covariance_matrix (2,2) / (n - 1);
	  cmat (0,0) = f21;
	  cmat (0,1) = f22;
	  cmat (0,2) = f23;
	  cmat (1,0) = f22;
	  cmat (1,1) = f24;
	  cmat (1,2) = f25;
	  cmat (2,0) = f23;
	  cmat (2,1) = f25;
	  cmat (2,2) = f26;
	  // 固有値を求める
	  EigenSolver<MatrixXf> es(cmat, false);
	  complex<double> lambda1 = es.eigenvalues()[0];
	  complex<double> lambda2 = es.eigenvalues()[1];
	  complex<double> lambda3 = es.eigenvalues()[2];
	  v11 = lambda1.real();
	  v22 = lambda2.real();
	  v33 = lambda3.real();
	  std::vector<double> v;
	  v.push_back (v11);
	  v.push_back (v22);
	  v.push_back (v33);
	  std::sort(v.begin(), v.end() );
	  v3 = v[0];
	  v2 = v[1];
	  v1 = v[2];
	  v.clear();
	  double nnn = v1 * v2 * v3;
	  f31 = (v1 - v2) / v1;
	  f32 = (v2 - v3) / v1;
	  f33 = v3 / v1;
	  f34 = pow(nnn, 1.0 / 3.0);
	  f35 = (v1 - v3) / v1;
	  f36 = -(v1 * log(v1) + v2 * log(v2) + v3 * log(v3));
	  f37 = v3 / (v1 + v2 + v3);
	  // 特徴ベクトルの生成
	  std::vector<float> features;
	  features.push_back(f21);
	  features.push_back(f22);
	  features.push_back(f23);
	  features.push_back(f24);
	  features.push_back(f25);
	  features.push_back(f26);
	  features.push_back(f31);
	  features.push_back(f32);
	  features.push_back(f33);
	  features.push_back(f34);
	  features.push_back(f35);
	  features.push_back(f36);
	  features.push_back(f37);
	  // 識別
	  for(int j=0; j<13; j++) {
		sample.at<float>(0,j) = features[j];
	  }
	  float res = svm.predict(sample);

	  sprintf (Box1, "Box1.%i", i);
	  if (res == 1) {
		viewer.addCube (minp1.x, maxp1.x, minp1.y, maxp1.y, minp1.z, maxp1.z, 0.0, 1.0, 0.0, Box1, V1);			
	  } else {
		viewer.addCube (minp1.x, maxp1.x, minp1.y, maxp1.y, minp1.z, maxp1.z, 1.0, 0.0, 0.0, Box1, V1);
	  }
	}
  }
  cout << "クラスタリング2の処理時間：" << time.toc () << "[ms]" << endl;

  // クラスタリング(k1_cloud)
  // 30m以内
  time.tic ();
  tree->setInputCloud (filter_cloud2);
  ec.setClusterTolerance (0.25);
  ec.setMinClusterSize (10);
  ec.setMaxClusterSize (100000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (filter_cloud2);
  ec.extract (cluster_indices2);
	
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices2.begin (); it != cluster_indices2.end (); ++it) {
	PointT minp1, maxp1;
	pcl::PointCloud<PointT>::Ptr cloud_cluster (new pcl::PointCloud<PointT>);

	for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
	  cloud_cluster->points.push_back (filter_cloud2->points[*pit]);		
	cloud_cluster->width = cloud_cluster->points.size ();
	cloud_cluster->height = 1;
	cloud_cluster->is_dense = true;

	feature_extractor.setInputCloud (cloud_cluster);
	feature_extractor.compute ();
	feature_extractor.getAABB (minp1, maxp1);
	Eigen::Vector3f dx(maxp1.x-minp1.x, 0.0, 0.0);
	Eigen::Vector3f dy(0.0, maxp1.y-minp1.y, 0.0);
	Eigen::Vector3f dz(0.0, 0.0, maxp1.z-minp1.z);
	f13 = dx.norm();
	f12 = dy.norm();
	f11 = dz.norm();

	if ((f11 >= 0.45)&&(f11 <= 2.0)&&(f12 <= 1.2)&&(f13 >= 0.05)&&(f13 <= 1.2)){
	  i1++;

	  cv::Mat sample(1, 13, CV_32FC1);

	  // 共分散行列と中心を求める
	  EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;
	  Eigen::Matrix3f cmat;
	  Eigen::Vector4f xyz_centroid;
	  pcl::compute3DCentroid (*cloud_cluster, xyz_centroid);
	  pcl::computeCovarianceMatrix (*cloud_cluster, xyz_centroid, covariance_matrix);
	  n = cloud_cluster->points.size();
	  f21 = covariance_matrix (0,0) / (n - 1);
	  f22 = covariance_matrix (0,1) / (n - 1);
	  f23 = covariance_matrix (0,2) / (n - 1);
	  f24 = covariance_matrix (1,1) / (n - 1);
	  f25 = covariance_matrix (1,2) / (n - 1);
	  f26 = covariance_matrix (2,2) / (n - 1);
	  cmat (0,0) = f21;
	  cmat (0,1) = f22;
	  cmat (0,2) = f23;
	  cmat (1,0) = f22;
	  cmat (1,1) = f24;
	  cmat (1,2) = f25;
	  cmat (2,0) = f23;
	  cmat (2,1) = f25;
	  cmat (2,2) = f26;
	  // 固有値を求める
	  EigenSolver<MatrixXf> es(cmat, false);
	  complex<double> lambda1 = es.eigenvalues()[0];
	  complex<double> lambda2 = es.eigenvalues()[1];
	  complex<double> lambda3 = es.eigenvalues()[2];
	  v11 = lambda1.real();
	  v22 = lambda2.real();
	  v33 = lambda3.real();
	  std::vector<double> v;
	  v.push_back (v11);
	  v.push_back (v22);
	  v.push_back (v33);
	  std::sort(v.begin(), v.end() );
	  v3 = v[0];
	  v2 = v[1];
	  v1 = v[2];
	  v.clear();
	  double nnn = v1 * v2 * v3;
	  f31 = (v1 - v2) / v1;
	  f32 = (v2 - v3) / v1;
	  f33 = v3 / v1;
	  f34 = pow(nnn, 1.0 / 3.0);
	  f35 = (v1 - v3) / v1;
	  f36 = -(v1 * log(v1) + v2 * log(v2) + v3 * log(v3));
	  f37 = v3 / (v1 + v2 + v3);
	  // 特徴ベクトルの生成
	  std::vector<float> features;
	  features.push_back(f21);
	  features.push_back(f22);
	  features.push_back(f23);
	  features.push_back(f24);
	  features.push_back(f25);
	  features.push_back(f26);
	  features.push_back(f31);
	  features.push_back(f32);
	  features.push_back(f33);
	  features.push_back(f34);
	  features.push_back(f35);
	  features.push_back(f36);
	  features.push_back(f37);
	  // 識別
	  for(int j=0; j<13; j++) {
		sample.at<float>(0,j) = features[j];
	  }
	  float res = svm.predict(sample);
	  
	  sprintf (Box2, "Box2.%i", i1);
	  if (res == 1) {
		viewer.addCube (minp1.x, maxp1.x, minp1.y, maxp1.y, minp1.z, maxp1.z, 0.0, 1.0, 0.0, Box2, V2);
	  } else {
		viewer.addCube (minp1.x, maxp1.x, minp1.y, maxp1.y, minp1.z, maxp1.z, 1.0, 0.0, 0.0, Box2, V2);
	  }
	}
  }
  cout << "クラスタリング3の処理時間：" << time.toc () << "[ms]" << endl;

  // 30m以上
  time.tic ();
  tree->setInputCloud (filter_cloud3);
  ec.setClusterTolerance (0.6);
  ec.setMinClusterSize (3);
  ec.setMaxClusterSize (100000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (filter_cloud3);
  ec.extract (cluster_indices3);
  
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices3.begin (); it != cluster_indices3.end (); ++it) {
	PointT minp1, maxp1;
	pcl::PointCloud<PointT>::Ptr cloud_cluster (new pcl::PointCloud<PointT>);

	for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
	  cloud_cluster->points.push_back (filter_cloud3->points[*pit]);		
	cloud_cluster->width = cloud_cluster->points.size ();
	cloud_cluster->height = 1;
	cloud_cluster->is_dense = true;

	feature_extractor.setInputCloud (cloud_cluster);
	feature_extractor.compute ();
	feature_extractor.getAABB (minp1, maxp1);
	Eigen::Vector3f dx(maxp1.x-minp1.x, 0.0, 0.0);
	Eigen::Vector3f dy(0.0, maxp1.y-minp1.y, 0.0);
	Eigen::Vector3f dz(0.0, 0.0, maxp1.z-minp1.z);
	f13 = dx.norm();
	f12 = dy.norm();
	f11 = dz.norm();

	if ((f11 >= 0.45)&&(f11 <= 2.0)&&(f12 <= 1.2)&&(f13 >= 0.05)&&(f13 <= 1.2)){
	  i1++;
	  sprintf (name2, "2.%i", i3);
	  viewer.addText3D (name2, cloud_cluster->points[0], 0.25, 1.0, 1.0);
	  i3++;

	  y_k1.push_back(f13);
	  z_k1.push_back(f11);
	  minp_h_k1.push_back(minp1);
	  maxp_h_k1.push_back(maxp1);
	  human_k1.push_back(cloud_cluster);

	  cv::Mat sample(1, 13, CV_32FC1);

	  // 共分散行列と中心を求める
	  EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;
	  Eigen::Matrix3f cmat;
	  Eigen::Vector4f xyz_centroid;
	  pcl::compute3DCentroid (*cloud_cluster, xyz_centroid);
	  pcl::computeCovarianceMatrix (*cloud_cluster, xyz_centroid, covariance_matrix);
	  n = cloud_cluster->points.size();
	  f21 = covariance_matrix (0,0) / (n - 1);
	  f22 = covariance_matrix (0,1) / (n - 1);
	  f23 = covariance_matrix (0,2) / (n - 1);
	  f24 = covariance_matrix (1,1) / (n - 1);
	  f25 = covariance_matrix (1,2) / (n - 1);
	  f26 = covariance_matrix (2,2) / (n - 1);
	  cmat (0,0) = f21;
	  cmat (0,1) = f22;
	  cmat (0,2) = f23;
	  cmat (1,0) = f22;
	  cmat (1,1) = f24;
	  cmat (1,2) = f25;
	  cmat (2,0) = f23;
	  cmat (2,1) = f25;
	  cmat (2,2) = f26;
	  // 固有値を求める
	  EigenSolver<MatrixXf> es(cmat, false);
	  complex<double> lambda1 = es.eigenvalues()[0];
	  complex<double> lambda2 = es.eigenvalues()[1];
	  complex<double> lambda3 = es.eigenvalues()[2];
	  v11 = lambda1.real();
	  v22 = lambda2.real();
	  v33 = lambda3.real();
	  std::vector<double> v;
	  v.push_back (v11);
	  v.push_back (v22);
	  v.push_back (v33);
	  std::sort(v.begin(), v.end() );
	  v3 = v[0];
	  v2 = v[1];
	  v1 = v[2];
	  v.clear();
	  double nnn = v1 * v2 * v3;
	  f31 = (v1 - v2) / v1;
	  f32 = (v2 - v3) / v1;
	  f33 = v3 / v1;
	  f34 = pow(nnn, 1.0 / 3.0);
	  f35 = (v1 - v3) / v1;
	  f36 = -(v1 * log(v1) + v2 * log(v2) + v3 * log(v3));
	  f37 = v3 / (v1 + v2 + v3);
	  // 特徴ベクトルの生成
	  std::vector<float> features;
	  features.push_back(f21);
	  features.push_back(f22);
	  features.push_back(f23);
	  features.push_back(f24);
	  features.push_back(f25);
	  features.push_back(f26);
	  features.push_back(f31);
	  features.push_back(f32);
	  features.push_back(f33);
	  features.push_back(f34);
	  features.push_back(f35);
	  features.push_back(f36);
	  features.push_back(f37);
	  // 識別
	  for(int j=0; j<13; j++) {
		sample.at<float>(0,j) = features[j];
	  }
	  float res = svm.predict(sample);

	  sprintf (Box2, "Box2.%i", i1);
	  if (res == 1) {
		viewer.addCube (minp1.x, maxp1.x, minp1.y, maxp1.y, minp1.z, maxp1.z, 0.0, 1.0, 0.0, Box2, V2);
	  } else {
		viewer.addCube (minp1.x, maxp1.x, minp1.y, maxp1.y, minp1.z, maxp1.z, 1.0, 0.0, 0.0, Box2, V2);
	  }
	}
  }
  cout << "クラスタリング4の処理時間：" << time.toc () << "[ms]" << endl;

  // クラスタリング(k2_cloud)
  // 30m以内
  time.tic ();
  tree->setInputCloud (filter_cloud4);
  ec.setClusterTolerance (0.25);
  ec.setMinClusterSize (10);
  ec.setMaxClusterSize (100000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (filter_cloud4);
  ec.extract (cluster_indices4);
  
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices4.begin (); it != cluster_indices4.end (); ++it) {
	PointT minp1, maxp1;
	pcl::PointCloud<PointT>::Ptr cloud_cluster (new pcl::PointCloud<PointT>);

	for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
	  cloud_cluster->points.push_back (filter_cloud4->points[*pit]);		
	cloud_cluster->width = cloud_cluster->points.size ();
	cloud_cluster->height = 1;
	cloud_cluster->is_dense = true;

	feature_extractor.setInputCloud (cloud_cluster);
	feature_extractor.compute ();
	feature_extractor.getAABB (minp1, maxp1);
	Eigen::Vector3f dx(maxp1.x-minp1.x, 0.0, 0.0);
	Eigen::Vector3f dy(0.0, maxp1.y-minp1.y, 0.0);
	Eigen::Vector3f dz(0.0, 0.0, maxp1.z-minp1.z);
	f13 = dx.norm();
	f12 = dy.norm();
	f11 = dz.norm();

	if ((f11 >= 0.45)&&(f11 <= 2.0)&&(f12 <= 1.2)&&(f13 >= 0.1)&&(f13 <= 1.2)){
	  i4++;

	  cv::Mat sample(1, 13, CV_32FC1);

	  // 共分散行列と中心を求める
	  EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;
	  Eigen::Matrix3f cmat;
	  Eigen::Vector4f xyz_centroid;
	  pcl::compute3DCentroid (*cloud_cluster, xyz_centroid);
	  pcl::computeCovarianceMatrix (*cloud_cluster, xyz_centroid, covariance_matrix);
	  n = cloud_cluster->points.size();
	  f21 = covariance_matrix (0,0) / (n - 1);
	  f22 = covariance_matrix (0,1) / (n - 1);
	  f23 = covariance_matrix (0,2) / (n - 1);
	  f24 = covariance_matrix (1,1) / (n - 1);
	  f25 = covariance_matrix (1,2) / (n - 1);
	  f26 = covariance_matrix (2,2) / (n - 1);
	  cmat (0,0) = f21;
	  cmat (0,1) = f22;
	  cmat (0,2) = f23;
	  cmat (1,0) = f22;
	  cmat (1,1) = f24;
	  cmat (1,2) = f25;
	  cmat (2,0) = f23;
	  cmat (2,1) = f25;
	  cmat (2,2) = f26;
	  // 固有値を求める
	  EigenSolver<MatrixXf> es(cmat, false);
	  complex<double> lambda1 = es.eigenvalues()[0];
	  complex<double> lambda2 = es.eigenvalues()[1];
	  complex<double> lambda3 = es.eigenvalues()[2];
	  v11 = lambda1.real();
	  v22 = lambda2.real();
	  v33 = lambda3.real();
	  std::vector<double> v;
	  v.push_back (v11);
	  v.push_back (v22);
	  v.push_back (v33);
	  std::sort(v.begin(), v.end() );
	  v3 = v[0];
	  v2 = v[1];
	  v1 = v[2];
	  v.clear();
	  double nnn = v1 * v2 * v3;
	  f31 = (v1 - v2) / v1;
	  f32 = (v2 - v3) / v1;
	  f33 = v3 / v1;
	  f34 = pow(nnn, 1.0 / 3.0);
	  f35 = (v1 - v3) / v1;
	  f36 = -(v1 * log(v1) + v2 * log(v2) + v3 * log(v3));
	  f37 = v3 / (v1 + v2 + v3);
	  // 特徴ベクトルの生成
	  std::vector<float> features;
	  features.push_back(f21);
	  features.push_back(f22);
	  features.push_back(f23);
	  features.push_back(f24);
	  features.push_back(f25);
	  features.push_back(f26);
	  features.push_back(f31);
	  features.push_back(f32);
	  features.push_back(f33);
	  features.push_back(f34);
	  features.push_back(f35);
	  features.push_back(f36);
	  features.push_back(f37);
	  // 識別
	  for(int j=0; j<13; j++) {
		sample.at<float>(0,j) = features[j];
	  }
	  float res = svm.predict(sample);
	
	  sprintf (Box3, "Box3.%i", i4);
	  if (res == 1) {
		viewer.addCube (minp1.x, maxp1.x, minp1.y, maxp1.y, minp1.z, maxp1.z, 0.0, 1.0, 0.0, Box3, V3);
	  } else {
		viewer.addCube (minp1.x, maxp1.x, minp1.y, maxp1.y, minp1.z, maxp1.z, 1.0, 0.0, 0.0, Box3, V3);
	  }
	}
  }
  cout << "クラスタリング5の処理時間：" << time.toc () << "[ms]" << endl;

  // 30m以上
  time.tic ();
  tree->setInputCloud (filter_cloud5);
  ec.setClusterTolerance (0.6);
  ec.setMinClusterSize (3);
  ec.setMaxClusterSize (100000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (filter_cloud5);
  ec.extract (cluster_indices5);
	
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices5.begin (); it != cluster_indices5.end (); ++it) {
	PointT minp1, maxp1;
	pcl::PointCloud<PointT>::Ptr cloud_cluster (new pcl::PointCloud<PointT>);

	for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
	  cloud_cluster->points.push_back (filter_cloud5->points[*pit]);		
	cloud_cluster->width = cloud_cluster->points.size ();
	cloud_cluster->height = 1;
	cloud_cluster->is_dense = true;

	feature_extractor.setInputCloud (cloud_cluster);
	feature_extractor.compute ();
	feature_extractor.getAABB (minp1, maxp1);
	Eigen::Vector3f dx(maxp1.x-minp1.x, 0.0, 0.0);
	Eigen::Vector3f dy(0.0, maxp1.y-minp1.y, 0.0);
	Eigen::Vector3f dz(0.0, 0.0, maxp1.z-minp1.z);
	f13 = dx.norm();
	f12 = dy.norm();
	f11 = dz.norm();

	if ((f11 >= 0.45)&&(f11 <= 2.0)&&(f12 <= 1.2)&&(f13 >= 0.05)&&(f13 <= 1.2)){
	  i4++;
	  sprintf (name3, "3.%i", i5);
	  viewer.addText3D (name3, cloud_cluster->points[0], 0.25, 1.0, 1.0);
	  i5++;
	  
	  y_k2.push_back(f13);
	  z_k2.push_back(f11);
	  minp_h_k2.push_back(minp1);
	  maxp_h_k2.push_back(maxp1);
	  human_k2.push_back(cloud_cluster);
	  
	  cv::Mat sample(1, 13, CV_32FC1);

	  // 共分散行列と中心を求める
	  EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;
	  Eigen::Matrix3f cmat;
	  Eigen::Vector4f xyz_centroid;
	  pcl::compute3DCentroid (*cloud_cluster, xyz_centroid);
	  pcl::computeCovarianceMatrix (*cloud_cluster, xyz_centroid, covariance_matrix);
	  n = cloud_cluster->points.size();
	  f21 = covariance_matrix (0,0) / (n - 1);
	  f22 = covariance_matrix (0,1) / (n - 1);
	  f23 = covariance_matrix (0,2) / (n - 1);
	  f24 = covariance_matrix (1,1) / (n - 1);
	  f25 = covariance_matrix (1,2) / (n - 1);
	  f26 = covariance_matrix (2,2) / (n - 1);
	  cmat (0,0) = f21;
	  cmat (0,1) = f22;
	  cmat (0,2) = f23;
	  cmat (1,0) = f22;
	  cmat (1,1) = f24;
	  cmat (1,2) = f25;
	  cmat (2,0) = f23;
	  cmat (2,1) = f25;
	  cmat (2,2) = f26;
	  // 固有値を求める
	  EigenSolver<MatrixXf> es(cmat, false);
	  complex<double> lambda1 = es.eigenvalues()[0];
	  complex<double> lambda2 = es.eigenvalues()[1];
	  complex<double> lambda3 = es.eigenvalues()[2];
	  v11 = lambda1.real();
	  v22 = lambda2.real();
	  v33 = lambda3.real();
	  std::vector<double> v;
	  v.push_back (v11);
	  v.push_back (v22);
	  v.push_back (v33);
	  std::sort(v.begin(), v.end() );
	  v3 = v[0];
	  v2 = v[1];
	  v1 = v[2];
	  v.clear();
	  double nnn = v1 * v2 * v3;
	  f31 = (v1 - v2) / v1;
	  f32 = (v2 - v3) / v1;
	  f33 = v3 / v1;
	  f34 = pow(nnn, 1.0 / 3.0);
	  f35 = (v1 - v3) / v1;
	  f36 = -(v1 * log(v1) + v2 * log(v2) + v3 * log(v3));
	  f37 = v3 / (v1 + v2 + v3);
	  // 特徴ベクトルの生成
	  std::vector<float> features;
	  features.push_back(f21);
	  features.push_back(f22);
	  features.push_back(f23);
	  features.push_back(f24);
	  features.push_back(f25);
	  features.push_back(f26);
	  features.push_back(f31);
	  features.push_back(f32);
	  features.push_back(f33);
	  features.push_back(f34);
	  features.push_back(f35);
	  features.push_back(f36);
	  features.push_back(f37);
	  // 識別
	  for(int j=0; j<13; j++) {
		sample.at<float>(0,j) = features[j];
	  }
	  float res = svm.predict(sample);

	  sprintf (Box3, "Box3.%i", i4);
	  if (res == 1) {
		viewer.addCube (minp1.x, maxp1.x, minp1.y, maxp1.y, minp1.z, maxp1.z, 0.0, 1.0, 0.0, Box3, V3);
	  } else {
		viewer.addCube (minp1.x, maxp1.x, minp1.y, maxp1.y, minp1.z, maxp1.z, 1.0, 0.0, 0.0, Box3, V3);
	  }
	}
  }
  cout << "クラスタリング6の処理時間：" << time.toc () << "[ms]" << endl;

  // 重畳
  time.tic ();
  for (int C = 0; C < human_k2.size (); C++){
	float centerx_k2;
	float centery_k2;
	float centerz_k2;
	centerx_k2 = ((minp_h_k2[C].x-maxp_h_k2[C].x)/2.0) + maxp_h_k2[C].x;
	centery_k2 = ((minp_h_k2[C].y-maxp_h_k2[C].y)/2.0) + maxp_h_k2[C].y;
	centerz_k2 = ((minp_h_k2[C].z-maxp_h_k2[C].z)/2.0) + maxp_h_k2[C].z;
	
	for (int A = 0; A < human_k1.size (); A++){
	  if ((0.7*z_k1[A] <= z_k2[C])&&(z_k2[C] <= 1.4*z_k1[A])&&(0.45*y_k1[A] <= y_k2[C])&&(y_k2[C] <= 2.2*y_k1[A])){// k+2とk+1の比較（大きさ）
		float centerx_k1;
		float centery_k1;
		float centerz_k1;
		centerx_k1 = ((minp_h_k1[A].x-maxp_h_k1[A].x)/2.0) + maxp_h_k1[A].x;
		centery_k1 = ((minp_h_k1[A].y-maxp_h_k1[A].y)/2.0) + maxp_h_k1[A].y;
		centerz_k1 = ((minp_h_k1[A].z-maxp_h_k1[A].z)/2.0) + maxp_h_k1[A].z;
		if ((pow(centerx_k2-centerx_k1,2.0)+pow(centery_k2-centery_k1,2.0)) <= 1.0){// k+2とk+1の比較（位置）

		  for (int B = 0; B < human_k.size (); B++){
			if ((0.7*z_k[B] <= z_k1[A])&&(z_k1[A] <= 1.4*z_k[B])&&(0.45*y_k[B] <= y_k1[A])&&(y_k1[A] <= 2.2*y_k[B])){// kとk+1の比較（大きさ）
			  float centerx_k;
			  float centery_k;
			  float centerz_k;
			  centerx_k = ((minp_h_k[B].x-maxp_h_k[B].x)/2.0) + maxp_h_k[B].x;
			  centery_k = ((minp_h_k[B].y-maxp_h_k[B].y)/2.0) + maxp_h_k[B].y;
			  centerz_k = ((minp_h_k[B].z-maxp_h_k[B].z)/2.0) + maxp_h_k[B].z;
			  if ((pow(centerx_k1-centerx_k,2.0)+pow(centery_k1-centery_k,2.0)) <= 1.0){// kとk+1の比較（位置）
				cout << B << "と" << A << "と" << C << "が一致" << endl;
				pcl::PointCloud<PointT>::Ptr transform (new pcl::PointCloud<PointT>);
				pcl::PointCloud<PointT>::Ptr transform1 (new pcl::PointCloud<PointT>);
				pcl::PointCloud<PointT>::Ptr super_cloud (new pcl::PointCloud<PointT>);
				// kフレームの人物点群をk+2の人物点群に重なるような変換行列の設定
				transform_1 (0,3) = centerx_k2 - centerx_k;
				transform_1 (1,3) = centery_k2 - centery_k;
				transform_1 (2,3) = centerz_k2 - centerz_k;
				// k+1フレームの人物点群をk+2の人物点群に重なるような変換行列の設定
				transform_2 (0,3) = centerx_k2 - centerx_k1;
				transform_2 (1,3) = centery_k2 - centery_k1;
				transform_2 (2,3) = centerz_k2 - centerz_k1;
				// transform
				pcl::transformPointCloud (*human_k[B], *transform, transform_1);
				pcl::transformPointCloud (*human_k1[A], *transform1, transform_2);
				// super_cloudに重畳された点群が格納
				*super_cloud = *human_k2[C] + *transform;
				*super_cloud = *super_cloud + *transform1;
				sprintf (supercloud, "cloud%i-%i-%i", A, B, C);
				// 重畳された点群をviewportに表示
				viewer.addPointCloud<PointT> (super_cloud, handler,supercloud, V4); 

				PointT minp1, maxp1;
				feature_extractor.setInputCloud (super_cloud);
				feature_extractor.compute ();
				feature_extractor.getAABB (minp1, maxp1);

				cv::Mat sample(1, 13, CV_32FC1);

				// 共分散行列と中心を求める
				EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;
				Eigen::Matrix3f cmat;
				Eigen::Vector4f xyz_centroid;
				pcl::compute3DCentroid (*super_cloud, xyz_centroid);
				pcl::computeCovarianceMatrix (*super_cloud, xyz_centroid, covariance_matrix);
				n = super_cloud->points.size();
				f21 = covariance_matrix (0,0) / (n - 1);
				f22 = covariance_matrix (0,1) / (n - 1);
				f23 = covariance_matrix (0,2) / (n - 1);
				f24 = covariance_matrix (1,1) / (n - 1);
				f25 = covariance_matrix (1,2) / (n - 1);
				f26 = covariance_matrix (2,2) / (n - 1);
				cmat (0,0) = f21;
				cmat (0,1) = f22;
				cmat (0,2) = f23;
				cmat (1,0) = f22;
				cmat (1,1) = f24;
				cmat (1,2) = f25;
				cmat (2,0) = f23;
				cmat (2,1) = f25;
				cmat (2,2) = f26;
				// 固有値を求める
				EigenSolver<MatrixXf> es(cmat, false);
				complex<double> lambda1 = es.eigenvalues()[0];
				complex<double> lambda2 = es.eigenvalues()[1];
				complex<double> lambda3 = es.eigenvalues()[2];
				v11 = lambda1.real();
				v22 = lambda2.real();
				v33 = lambda3.real();
				std::vector<double> v;
				v.push_back (v11);
				v.push_back (v22);
				v.push_back (v33);
				std::sort(v.begin(), v.end() );
				v3 = v[0];
				v2 = v[1];
				v1 = v[2];
				v.clear();
				double nnn = v1 * v2 * v3;
				f31 = (v1 - v2) / v1;
				f32 = (v2 - v3) / v1;
				f33 = v3 / v1;
				f34 = pow(nnn, 1.0 / 3.0);
				f35 = (v1 - v3) / v1;
				f36 = -(v1 * log(v1) + v2 * log(v2) + v3 * log(v3));
				f37 = v3 / (v1 + v2 + v3);
				// 特徴ベクトルの生成
				std::vector<float> features;
				features.push_back(f21);
				features.push_back(f22);
				features.push_back(f23);
				features.push_back(f24);
				features.push_back(f25);
				features.push_back(f26);
				features.push_back(f31);
				features.push_back(f32);
				features.push_back(f33);
				features.push_back(f34);
				features.push_back(f35);
				features.push_back(f36);
				features.push_back(f37);
				// 識別
				for(int j=0; j<13; j++) {
				  sample.at<float>(0,j) = features[j];
				}
				float res = svm.predict(sample);

				sprintf (superbox, "superbox%i-%i-%i", A, B, C);
				if (res == 1) {
				  viewer.addCube (minp1.x, maxp1.x, minp1.y, maxp1.y, minp1.z, maxp1.z, 0.0, 1.0, 0.0, superbox, V4);
				} else {
				  viewer.addCube (minp1.x, maxp1.x, minp1.y, maxp1.y, minp1.z, maxp1.z, 1.0, 0.0, 0.0, superbox, V4);
				}
			  }
			}
		  }
		}
	  }
	}
  }
  cout << "重畳処理の処理時間：" << time.toc () << "[ms]" << endl;

  //viewer.setCameraPosition(-0.46661,37.9138,1.52971,-0.456605,38.0134,1.51297,0.0198642,0.163714,0.986308);
  viewer.setCameraPosition(-6.38991,-28.669,12.8896,-0.946912,0.188503,3.23513,0.0690496,0.304779,0.949917);
  //viewer.setCameraPosition(6.27088,59.0348,48.3711,-3.65551,52.728,-30.941,-0.204204,-0.973498,0.102968);
  viewer.setSize (1280, 1024);
	
  while (!viewer.wasStopped())
    {
	  viewer.spinOnce ();
    }
	
  pcl::visualization::Camera getCamera;
  viewer.getCameraParameters( getCamera );
  cout << "camera parameter:" << endl;
  cout << getCamera.pos[0] << "," << getCamera.pos[1] << "," << getCamera.pos[2] << "," << endl
	   << getCamera.focal[0] << "," << getCamera.focal[1] << "," << getCamera.focal[2] << "," << endl
	   << getCamera.view[0] << "," << getCamera.view[1] << "," << getCamera.view[2]  << endl;
	
  return (0);
}
