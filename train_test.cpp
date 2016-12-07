#include "train_test.h"
#include <QDir>
#include <QString>
#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>

train_test::train_test(const std::string &train_root, const std::string &data_root)
{
    m_train_root = train_root;
    m_test_root = "../test_data/";
    m_data_root = data_root;
    m_feature_size = 128;
    m_para_num = 6;
    m_casscade_sum = 7;
    load_3DMM_data();
    load_keypoints_id();
    //use sift feature
    m_feature_detector = new SIFTDectector;
}

void train_test::read_groundtruth_data()
{
    QDir Path(QString(m_train_root.data()));
    Path.setFilter(QDir::AllDirs);
    Path.setSorting(QDir::Name);
    QStringList file_list = Path.entryList();
    std::string name;
    m_train_individuals.clear();
    for(QStringList::Iterator iter = file_list.begin();iter!=file_list.end();iter++)
    {

        name = (*iter).toStdString();
        if(name=="."||name=="..")
            continue;
//        std::cout<<name<<std::endl;
        m_train_individuals.push_back(name);
    }
    m_groundtruth_shapes.resize(m_st_pc_num,m_train_individuals.size());
    m_groundtruth_exps.resize(m_expression_pc_num,m_train_individuals.size());

    m_train_shapes.resize(m_st_pc_num,m_train_individuals.size());
    m_train_exps.resize(m_expression_pc_num,m_train_individuals.size());

    m_train_individuals_datanum.clear();
    m_total_images=0;
    for(int i=0;i<m_train_individuals.size();i++)
    {
        std::string root = m_train_root+m_train_individuals[i]+"/";
        QDir path(QString(root.data()));
        path.setFilter(QDir::Files);
        QStringList filters("*.jpg");
        path.setNameFilters(filters);
        path.setSorting(QDir::Name);
        QStringList entrys = path.entryList();
        m_train_individuals_datanum.push_back(entrys.size());
        m_total_images+=entrys.size();
    }
    m_groundtruth_paras.resize(m_para_num, m_total_images);
    m_groundtruth_box.resize(4,m_total_images);
    m_train_paras.resize(m_para_num,m_total_images);
    m_visible_features.resize(m_keypoint_id.size()*m_feature_size,m_total_images);

    std::cout<<"train data has "<<m_train_individuals.size()<<" individuals."<<" total "<<m_total_images<<" images."<<std::endl;
    read_ground_shape_exp();
    read_ground_para_box();
    read_train_images();
}



void train_test::train_para_only()
{
    initial_x_train_para_only();
//    int casscade_num = m_casscade_sum;
    m_para_Rs.resize(m_casscade_sum, Eigen::MatrixXf());
    m_para_bs.resize(m_casscade_sum, Eigen::VectorXf());
    for(m_casscade_level=0; m_casscade_level<m_casscade_sum; m_casscade_level++)
    {
        compute_all_visible_features();
        compute_paras_R_b();
        update_para();
    }
}

void train_test::test_para_only()
{
    read_learned_para_Rbs();
    initial_x_train_para_only();
    for(m_casscade_level=0; m_casscade_level<m_para_Rs.size(); m_casscade_level++)
    {
        compute_all_visible_features();
        update_para();
    }
    save_test_result_imgs();
}

void train_test::save_para_result()
{
    FILE *file = fopen("../learn_result/para_Rs.bin","wb");
    int R_col = m_keypoint_id.size()*m_feature_size;
    int R_row = m_para_num;
    fwrite(&m_casscade_sum,sizeof(int),1,file);
    fwrite(&R_row,sizeof(int),1,file);
    fwrite(&R_col,sizeof(int),1,file);
    for(int i=0;i<m_casscade_sum;i++)
    {
        fwrite(m_para_Rs[i].data(),sizeof(float),R_col*R_row,file);
    }
    fclose(file);
    file = fopen("../learn_result/para_bs.bin","wb");
    fwrite(&m_casscade_sum,sizeof(int),1,file);
    fwrite(&m_para_num,sizeof(int),1,file);
    for(int i=0;i<m_casscade_sum;i++)
    {
        fwrite(m_para_bs[i].data(),sizeof(float),m_para_num,file);
    }
    fclose(file);
}

void train_test::initial_x_train_para_only()
{
    initial_shape_exp_with_groundtruth();
    initial_para();
}

void train_test::compute_all_visible_features()
{
    for(int p_id=0; p_id<m_train_individuals.size(); p_id++)
    {
        update_mv(p_id);
        for(int index=0;index<m_train_individuals_datanum[p_id];index++)
        {
            compute_visible_features(p_id,index);
        }
    }
}
// ||delta_x - R*vfeature-b||^2+lamda*||R||^2
void train_test::compute_paras_R_b()
{
    float lamda = 0.1;
    Eigen::MatrixXf delta_x(m_para_num, m_total_images);
    for(int i=0;i<m_total_images;i++)
    {
            delta_x.col(i) = m_groundtruth_paras.col(i) - m_train_paras.col(i);
    }
    //combine R and b to solve, new_R=(R|b)
//    const int R_row = m_para_num;
    const int R_col = m_keypoint_id.size()*m_feature_size+1;
    Eigen::MatrixXf lhs(R_col,R_col);
    Eigen::MatrixXf rhs(R_col,m_para_num);
    lhs.block(0,0,R_col-1,R_col-1).selfadjointView<Eigen::Upper>().rankUpdate(m_visible_features,1.0);
    lhs.block(0,R_col-1,R_col-1,1) = m_visible_features.rowwise().sum();
    lhs(R_col-1,R_col-1) = float(m_total_images);
    //add regular
    lhs.block(0,0,R_col-1,R_col-1)+=(lamda*Eigen::VectorXf(R_col-1).setOnes()).asDiagonal();

    rhs.block(0,0,R_col-1,m_para_num) = m_visible_features*delta_x.transpose();
    rhs.bottomRows(1) = delta_x.transpose().colwise().sum();


    std::cout<<"casscade para "<<m_casscade_level<<" compute for A("<<lhs.rows()<<"*"<<lhs.cols()<<")..."<<std::endl;

    Eigen::MatrixXf temp = lhs.selfadjointView<Eigen::Upper>().llt().solve(rhs);
    Eigen::MatrixXf result = temp.transpose();
    std::cout<<"done! "<<std::endl;
    std::cout<<"casscade para "<<m_casscade_level<<" result norm: "<<result.norm()<<"; sqrt energy is "<<(lhs*result.transpose()-rhs).norm()<<std::endl;
    Eigen::MatrixXf &R = m_para_Rs[m_casscade_level];
    R.resize(m_para_num,m_feature_size*m_keypoint_id.size());
    memcpy(R.data(),result.data(), sizeof(float)*R.size());
    Eigen::VectorXf &b = m_para_bs[m_casscade_level];
    b.resize(m_para_num);
    b = result.col(R_col-1);

}

void train_test::update_para()
{
    Eigen::MatrixXf delta_x(m_para_num, m_total_images);
    delta_x = (m_para_Rs[m_casscade_level]*m_visible_features).colwise()+m_para_bs[m_casscade_level];
    m_train_paras+=delta_x;
    Eigen::MatrixXf delta_para = m_groundtruth_paras - m_train_paras;
    Eigen::VectorXf delta_norm = delta_para.colwise().norm();
    std::cout<<"casscade para "<<m_casscade_level<<" para with ground truth norm: mean: "<<delta_norm.mean()
            <<" max: "<<delta_norm.maxCoeff()<<" min: "<<delta_norm.minCoeff()<<std::endl;

}

void train_test::update_mv(int individual)
{
    Eigen::VectorXf shape = m_train_shapes.col(individual);
    Eigen::VectorXf exp = m_train_exps.col(individual);
    m_v.resize(m_v.size(),1);
    m_mean_face.resize(m_mean_face.size(),1);
    m_v = m_mean_face + m_shape_pc*shape + m_expression_pc*exp;
}

void train_test::compute_visible_features(int person, int img)
{
    int col = getDataColId(person,img);
    float *para = m_train_paras.col(col).data();
    float scale = para[0];
    float ax = para[1]; float ay = para[2]; float az = para[3];
    float tx = para[4]; float ty = para[5];

    Eigen::Affine3f transformation;
    transformation  = Eigen::AngleAxisf(ax, Eigen::Vector3f::UnitX()) *
                      Eigen::AngleAxisf(ay, Eigen::Vector3f::UnitY()) *
                      Eigen::AngleAxisf(az, Eigen::Vector3f::UnitZ());
    Eigen::Matrix3f R = transformation.rotation();
    m_v.resize(3,m_v.size()/3);
    Eigen::MatrixXf verts(3, m_keypoint_id.size());
    for(int i=0;i<m_keypoint_id.size();i++)
    {
        verts.col(i) = m_v.col(m_keypoint_id[i]);
    }
    Eigen::MatrixXf feature_pos(2,m_keypoint_id.size()) ;
    verts = R*verts;
    feature_pos.row(0) = verts.row(0);
    feature_pos.row(1) = verts.row(1);
    feature_pos *= scale;
    Eigen::Vector2f trans;  trans(0) = tx;  trans(1) = ty;
    feature_pos.colwise() += trans;

    cv::Mat &image = m_train_imgs[col];
//    cv::Mat grayImage;
//    if(image.channels()==3)
//        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
//    else
//        grayImage = image;


    //visibles compute; can change to use openGL, more accurate
    std::vector<bool>   visuals;
    compute_keypoint_visible(R*m_v,visuals);
    // add extract feature code, para: grayImage feature_pos visuals, result:  visible_features; need to compute visible
    Eigen::VectorXf visible_features(m_feature_size*m_keypoint_id.size());
    Eigen::VectorXf scales(m_keypoint_id.size()); scales.setOnes();
    scales *= 6.0;
    m_feature_detector->DescriptorOnCustomPoints(image,feature_pos,scales,visible_features) ;
    for(int i=0;i<m_keypoint_id.size();i++)
    {
        if(!visuals[i])  //now, set unvisible feature to zero
        {
            for(int j=i*m_feature_size; j<(i+1)*m_feature_size; j++)
                visible_features(j) = 0.0;
        }
    }
    //**************************
    m_visible_features.col(col) = visible_features;
}

void train_test::compute_keypoint_visible(const Eigen::MatrixXf &verts, std::vector<bool> &visuals)
{
    float *addrv = m_mesh.point(m_mesh.vertices_begin().handle()).data();
    memcpy(addrv, verts.data(), sizeof(float)*verts.size());
//    m_mesh.update_normals();
    //for save compute time
    update_keypoints_face_normals(m_mesh,m_keypoint_id);
    visuals.resize(m_keypoint_id.size(), false);
    TriMesh::Normal zdir(0.0,0.0,1.0);
    for(int i=0; i< m_keypoint_id.size(); i++)
    {
        TriMesh::Normal normal = m_mesh.normal(TriMesh::VertexHandle(m_keypoint_id[i]));
//        Eigen::Vector3f normal = Eigen::Map<Eigen::Vector3f>(tnormal.data());
        float val = zdir|normal;
        if(val<0.0)
            visuals[i] = true;
    }
}

void train_test::read_learned_para_Rbs()
{
    std::string root="../learn_result/";
    FILE *file = fopen(std::string(root+"para_Rs.bin").data(),"rb");
    int num,row,col;
    fread(&num, sizeof(int), 1, file);
    fread(&row, sizeof(int), 1, file);
    fread(&col, sizeof(int), 1, file);
    m_para_Rs.clear();
    Eigen::MatrixXf R(row,col);
    for(int i=0 ;i<num; i++)
    {
        fread(R.data(), sizeof(float), row*col, file);
        m_para_Rs.push_back(R);
    }
    fclose(file);
    file = fopen((root+"para_bs.bin").data(), "rb");
    fread(&num, sizeof(int), 1, file);
    fread(&row, sizeof(int), 1, file);
    m_para_bs.clear();
    Eigen::VectorXf b(row);
    for(int i=0; i<num; i++)
    {
        fread(b.data(), sizeof(float), row, file);
        m_para_bs.push_back(b);
    }
    fclose(file);
}

void train_test::read_train_images()
{
    std::string file_name_base = "img_";
    m_train_imgs.clear();
    for(int i=0;i<m_train_individuals.size();i++)
    {
        std::string root = m_train_root+m_train_individuals[i]+"/";
        for(int id=0;id<m_train_individuals_datanum[i];id++)
        {
            QString num;    num.setNum(id);
            std::string name = root+file_name_base+num.toStdString()+".jpg";
            cv::Mat tmp=cv::imread(name,cv::IMREAD_GRAYSCALE);  //read as CV_32F type, 0~255.0;
            cv::Mat img;
            tmp.convertTo(img,CV_32F,1.0);
            m_train_imgs.push_back(img);
        }
    }
}

void train_test::read_ground_shape_exp()
{
    std::string name;
    float*  shapes = m_groundtruth_shapes.data();
    float*  exps = m_groundtruth_exps.data();
    for(int i=0;i<m_train_individuals.size();i++)
    {
        name = "/mesh_"+m_train_individuals[i]+".txt";
        std::ifstream file((m_train_root+m_train_individuals[i]+name).data());
        for(int i=0;i<m_st_pc_num;i++)
        {
            file>>(*shapes);
            shapes++;
        }
        for(int i=0;i<m_expression_pc_num;i++)
        {
            file>>(*exps);
            exps++;
        }
        file.close();
    }
}

void train_test::read_ground_para_box()
{
    std::string file_name_base = "img_";
    std::string box_name_base = "img_box_";
    float *data = m_groundtruth_paras.data();
    float *box_data = m_groundtruth_box.data();
    for(int i=0;i<m_train_individuals.size();i++)
    {
        std::string root = m_train_root+m_train_individuals[i]+"/";
        for(int id=0;id<m_train_individuals_datanum[i];id++)
        {
            QString num;    num.setNum(id);
            std::string name = root+file_name_base+num.toStdString()+".txt";
            std::string box_name = root+box_name_base+num.toStdString()+".txt";
            std::ifstream file(name.data());
            file>>(*data);  data++; //scale;
            float angle;    //angles, save arc
            file>>angle;    angle*=M_PI/180.0;  (*data) = angle;    data++;
            file>>angle;    angle*=M_PI/180.0;  (*data) = angle;    data++;
            file>>angle;    angle*=M_PI/180.0;  (*data) = angle;    data++;
            file>>(*data);  data++;
            file>>(*data);  data++;
            file.close();

            file.open(box_name.data());
            file>>(*box_data);  box_data++;
            file>>(*box_data);  box_data++;
            file>>(*box_data);  box_data++;
            file>>(*box_data);  box_data++;
            file.close();
        }
    }
}

void train_test::load_3DMM_data()
{
    int v_num,pc_num;
    FILE* file;
    file = fopen((m_data_root+"mainShapePCA.bin").data(), "rb");
    fread(&v_num,sizeof(int), 1, file);
    fread(&pc_num,sizeof(int),1,file);
    m_vertex_num = v_num;
    m_st_pc_num = pc_num;
    m_mean_shape.resize(3*m_vertex_num,1);
    fread(m_mean_shape.data(),sizeof(float),3*m_vertex_num,file);
    m_shape_pc.resize(3*m_vertex_num,m_st_pc_num);
    fread(m_shape_pc.data(),sizeof(float),m_shape_pc.size(),file);
    m_shape_sd.resize(m_st_pc_num,1);
    fread(m_shape_sd.data(),sizeof(float),m_st_pc_num,file);
    fclose(file);


//    m_mean_texture.resize(3*m_vertex_num,1);
//    m_texture_pc.resize(3*m_vertex_num,m_st_pc_num);
//    m_texture_sd.resize(m_st_pc_num,1);

//    file = fopen("../Data/mainTexturePCA.bin","rb");
//    fread(&v_num,sizeof(int), 1, file);
//    fread(&pc_num,sizeof(int),1,file);
//    fread(m_mean_texture.data(),sizeof(float),m_mean_texture.size(),file);
//    fread(m_texture_pc.data(),sizeof(float),m_texture_pc.size(),file);
//    fread(m_texture_sd.data(),sizeof(float),m_texture_sd.size(),file);
//    fclose(file);


    file = fopen((m_data_root+"DeltaExpPCA.bin").data(),"rb");
    fread(&v_num,sizeof(int), 1, file);
    fread(&pc_num,sizeof(int),1,file);
    m_expression_pc_num = pc_num;
    m_mean_expression.resize(3*m_vertex_num,1);
    m_expression_pc.resize(3*m_vertex_num,m_expression_pc_num);
    m_expression_sd.resize(m_expression_pc_num,1);

    fread(m_mean_expression.data(),sizeof(float),m_mean_expression.size(),file);
    fread(m_expression_pc.data(),sizeof(float),m_expression_pc.size(),file);
    fread(m_expression_sd.data(),sizeof(float),m_expression_sd.size(),file);
    fclose(file);

    m_mean_face.resize(3*m_vertex_num,1);
    m_v.resize(3,m_vertex_num); m_v.setZero();
    m_mean_face = m_mean_shape+m_mean_expression;


    file = fopen((m_data_root+"tri.bin").data(),"rb");
    fread(&m_face_num,sizeof(int),1,file);
    Eigen::MatrixXi triangles(3,m_face_num);
    fread(triangles.data(),sizeof(int),triangles.size(),file);
    fclose(file);

//    m_shape.resize(m_st_pc_num);    m_shape.setZero();
//    m_texture.resize(m_st_pc_num);  m_texture.setZero();
//    m_expression.resize(m_expression_pc_num);   m_expression.setZero();
////    m_expression.resize(m_expression_pc_num);   m_expression.setZero();
//    m_v.resize(3,m_vertex_num);
//    m_colors.resize(3,m_vertex_num);
    m_mesh.request_face_normals();
    m_mesh.request_vertex_normals();
//    m_mesh.request_vertex_colors();
    for(int i=0;i<m_vertex_num;i++)
        m_mesh.add_vertex(TriMesh::Point(0,0,0));
    for(int i=0;i<m_face_num;i++)
    {
        m_mesh.add_face(TriMesh::VertexHandle(triangles(0,i)),TriMesh::VertexHandle(triangles(2,i)),TriMesh::VertexHandle(triangles(1,i)));
    }

    std::cout<<"vertex num: "<<m_vertex_num<<"; face num: "<<m_face_num<<"; shape and texture pc num: "<<m_st_pc_num<<"; expression pc num: "<<m_expression_pc_num<<std::endl;
}

void train_test::load_keypoints_id()
{
    m_keypoint_id.clear();
    std::ifstream file((m_data_root+"keypoints_id.txt").data());
    if(!file.is_open())
    {
        std::cout<<"read key points id failed!"<<std::endl;
        return;
    }
    int num;
    file>>num;
    for(int i=0;i<num;i++)
    {
        int id;
        file>>id;
        m_keypoint_id.push_back(id);
    }
    file.close();
}

void train_test::initial_para()
{
    BOX box;
    compute_mean_shape_boundBox(box);
    for(int i=0;i<m_groundtruth_box.cols();i++)
    {
        float *data = m_groundtruth_box.col(i).data();
        float x = data[0];
        float y = data[1];
        float width = data[2];
        float height = data[3];
        float sca0 = float(width)/box.x_len;
        float sca1 = float(height)/box.y_len;
        float scale = (sca0+sca1)/2.0;
        float tx,ty;
        tx = float(x + width/2);
        ty = float(y + height/2);
        float *train_data=m_train_paras.col(i).data();
        train_data[0] = scale;
        train_data[1] = M_PI;
        train_data[2] = train_data[3] = 0.0;
        train_data[4] = tx;
        train_data[5] = ty;
    }
    Eigen::MatrixXf delta_para = m_groundtruth_paras - m_train_paras;
    Eigen::VectorXf delta_norm = delta_para.colwise().norm();
    std::cout<<"inital para with ground truth norm: mean: "<<delta_norm.mean()
            <<" max: "<<delta_norm.maxCoeff()<<" min: "<<delta_norm.minCoeff()<<std::endl;
}

void train_test::compute_mean_shape_boundBox(BOX &box)
{
    m_mean_face.resize(3,m_mean_face.size()/3);
    box.x_min=m_mean_face.row(0).minCoeff();   box.x_max=m_mean_face.row(0).maxCoeff();
    box.y_min=m_mean_face.row(1).minCoeff();   box.y_max=m_mean_face.row(1).maxCoeff();
    box.z_min=m_mean_face.row(2).minCoeff();   box.z_max=m_mean_face.row(2).maxCoeff();
    box.x_len =box.x_max-box.x_min;
    box.y_len = box.y_max-box.y_min;
    box.z_len = box.z_max-box.z_min;
}

void train_test::initial_shape_exp_with_groundtruth()
{
    memcpy(m_train_shapes.data(),m_groundtruth_shapes.data(),sizeof(float)*m_train_shapes.size());
    memcpy(m_train_exps.data(),m_groundtruth_exps.data(),sizeof(float)*m_train_exps.size());
}

void train_test::initial_shape_exp_with_mean()
{
    m_train_shapes.setZero();
    m_train_exps.setZero();
}

void train_test::save_test_result_imgs()
{
    std::string save_root = "../test_result/";
    for(int i=0; i<m_train_individuals.size(); i++)
    {
        for(int j=0; j<m_train_individuals_datanum[i]; j++)
        {
            int col = getDataColId(i,j);
            cv::Mat img = m_train_imgs[col].clone();
            update_mv(i);
            float *para = m_train_paras.col(col).data();
            float scale = para[0];
            float x,y,z;    x=para[1];  y=para[2];  z=para[3];
            float tx,ty;    tx = para[4];   ty = para[5];
            Eigen::Affine3f transformation;
            transformation  = Eigen::AngleAxisf(x, Eigen::Vector3f::UnitX()) *
                              Eigen::AngleAxisf(y, Eigen::Vector3f::UnitY()) *
                              Eigen::AngleAxisf(z, Eigen::Vector3f::UnitZ());
            Eigen::Matrix3f R = transformation.rotation();
            m_v.resize(3, m_v.size()/3);
            Eigen::MatrixXf temp = (scale*R*m_v).colwise()+Eigen::Vector3f(tx,ty,0.0);
            std::vector<bool>   visuals;
            compute_keypoint_visible(temp,visuals);
            for(int id=0;id<visuals.size();id++)
            {
                int vid = m_keypoint_id[id];
                Eigen::Vector3f vert = temp.col(vid);
                if(visuals[id])
                    cv::circle(img,cv::Point(vert(0),vert(1)),1,cv::Scalar(0,255,0),-1);
                else
                {
                    if(i<10)
                        cv::circle(img,cv::Point(vert(0),vert(1)),1,cv::Scalar(0,0,255),-1);
                }
            }
            QString iid;    iid.setNum(j);
            QDir save_dir(QString(save_root.data()));
            save_dir.mkdir(QString(m_train_individuals[i].data()));
            cv::imwrite(save_root+m_train_individuals[i]+"/"+iid.toStdString()+".jpg",img);
        }
    }
}

int train_test::getDataColId(int person, int img)
{
    int id=0;
    for(int i=0;i<person;i++)
        id += m_train_individuals_datanum[i];
    id+=img;
    return id;
}
//save some compute time!   do not use boundary points
void train_test::update_keypoints_face_normals(TriMesh &mesh, const std::vector<int> &ids)
{
    for(int id=0; id<ids.size(); id++)
    {
        TriMesh::VertexHandle vh(ids[id]);
        TriMesh::VertexFaceIter vfiter = mesh.vf_iter(vh);
        for(;vfiter.is_valid();vfiter++)
            mesh.set_normal(*vfiter,mesh.calc_face_normal(*vfiter));
        mesh.set_normal(vh,mesh.calc_vertex_normal(vh));
    }
}
