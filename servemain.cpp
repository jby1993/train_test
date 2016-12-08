#include "train_test.h"
#include <iostream>
int main(int argc, char *argv[])
{
    train_test  TrainTest("/data/jby/train_test/train_data/","/data/jby/train_test/Data/");
    //train on my synthesis imgs
//    TrainTest.set_train_root("/data/jby/train_test/train_data/");
//    TrainTest.set_data_root("/data/jby/train_test/Data/");
    std::cout<<"start read data!"<<std::endl;
    TrainTest.read_groundtruth_data();
    TrainTest.train_para_only();
    TrainTest.save_para_result();
    //test for my synthesis imgs
//    TrainTest.set_train_root("../test_data/");
//    TrainTest.read_groundtruth_data();
//    TrainTest.test_para_only();
    std::cout<<"done"<<std::endl;
    return 0;
}
