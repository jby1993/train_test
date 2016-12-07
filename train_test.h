#ifndef TRAIN_TEST_H
#define TRAIN_TEST_H
#include <Eigen/Dense>
#include <vector>
#include "tri_mesh.h"
#include <opencv/cv.h>
#include "siftdectector.h"
class train_test
{
public:
    train_test();
    void read_groundtruth_data();
    void set_train_root(const std::string &root){m_train_root = root;}
    void train_para_only();
    void test_para_only();
    void save_para_result();
private:
    void initial_x_train_para_only();
    void compute_all_visible_features();
    void compute_paras_R_b();
    void update_para();

    void update_mv(int individual);
    void compute_visible_features(int person, int img);
    void compute_keypoint_visible(const Eigen::MatrixXf &verts, std::vector<bool> &visuals);

    void read_learned_para_Rbs();

    void read_train_images();
    void read_ground_shape_exp();
    void read_ground_para_box();
    void load_3DMM_data();
    void load_keypoints_id();
    void initial_para();
    void compute_mean_shape_boundBox(BOX &box);
    void initial_shape_exp_with_groundtruth();
    void initial_shape_exp_with_mean();

    void save_test_result_imgs();
    int getDataColId(int person, int img);
    void update_keypoints_face_normals(TriMesh &mesh, const std::vector<int> &ids);
private:
    Eigen::MatrixXf m_groundtruth_paras; //angle has to be arc
    Eigen::MatrixXf m_groundtruth_shapes;
    Eigen::MatrixXf m_groundtruth_exps;
    Eigen::MatrixXf m_groundtruth_box;
    std::string m_train_root;
    std::string m_test_root;

    std::vector<cv::Mat>    m_train_imgs;   //save as CV_32F
    std::vector<std::string> m_train_individuals;
    std::vector<int>              m_train_individuals_datanum;
    std::vector<int>            m_keypoint_id;
    std::vector<Eigen::MatrixXf> m_para_Rs;
    std::vector<Eigen::VectorXf> m_para_bs;
    int m_feature_size;
    int m_casscade_level;
    int m_total_images;
    int m_para_num;
    int m_casscade_sum;

    Eigen::MatrixXf m_visible_features;
    Eigen::MatrixXf m_train_paras;
    Eigen::MatrixXf m_train_shapes;
    Eigen::MatrixXf m_train_exps;


    Eigen::MatrixXf m_shape_pc;
    Eigen::MatrixXf m_expression_pc;
    Eigen::MatrixXf m_mean_shape;
    Eigen::MatrixXf m_mean_expression;
    Eigen::MatrixXf m_mean_face;
    Eigen::VectorXf m_shape_sd;
    Eigen::VectorXf m_expression_sd;
    Eigen::MatrixXf m_v;
    TriMesh m_mesh;
    int m_vertex_num;
    int m_st_pc_num;
    int m_expression_pc_num;
    int m_face_num;

    FeatureDetector *m_feature_detector;
};

#endif // TRAIN_TEST_H
