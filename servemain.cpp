#include "train_test.h"
int main(int argc, char *argv[])
{
    train_test  TrainTest;
//    //train on my synthesis imgs
//    TrainTest.read_groundtruth_data();
//    TrainTest.train_para_only();
//    TrainTest.save_para_result();
    //test for my synthesis imgs
    TrainTest.set_train_root("../test_data/");
    TrainTest.read_groundtruth_data();
    TrainTest.test_para_only();
    std::cout<<"done"<<std::endl;
    return 0;
}
