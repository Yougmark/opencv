#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>
#include <thread>
#include <iomanip>
#include <stdexcept>
#include <opencv2/core/utility.hpp>
#include "opencv2/cudaobjdetect.hpp"
#include <opencv2/core/cuda.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "pgm.h"
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

VideoCapture vc;
pthread_barrier_t *init_barrier;
int hog_sample_errors;

__thread char hog_sample_errstr[80];

#define DEBUG 0
#define CheckError(e) \
do { int __ret = (e); \
if(__ret < 0) { \
    hog_sample_errors++; \
    char* errstr = strerror_r(errno, hog_sample_errstr, sizeof(errstr)); \
    fprintf(stderr, "%lu: Error %d (%s (%d)) @ %s:%s:%d\n",  \
            pthread_self(), __ret, errstr, errno, __FILE__, __FUNCTION__, __LINE__); \
}}while(0)
struct params_compute_hists
{
    int height;
    int width;
    cv::cuda::PtrStepSzf grad;
    cv::cuda::PtrStepSzb qangle;
    float* block_hists;
    cudaStream_t stream;
    cv::cuda::HOG* hog_agent;
    cv::cuda::GpuMat img;
    cv::cuda::GpuMat labels;
    std::vector<double>* confidences;
    std::vector<double>* all_confidences;
    double scale;
    cv::cuda::GpuMat smaller_img;
    cv::cuda::GpuMat* labels_array;
    cv::cuda::GpuMat* smaller_img_arr;
    cv::cuda::GpuMat* block_hists_arr;
    cv::cuda::GpuMat* grad_arr;
    cv::cuda::GpuMat* qangle_arr;
    std::vector<double> level_scale;
    std::vector<Rect>* found_locations;
    /* display */
    VideoCapture vc;
    Mat* img_to_show;
    cv::VideoWriter video_writer;
    vector<String> filenames;
    Mat* frame;
};

bool help_showed = false;

class Args
{
public:
    Args();
    static Args read(int argc, char** argv);

    string src;
    bool src_is_folder;
    bool src_is_video;
    bool src_is_camera;
    int camera_id;

    bool svm_load;
    string svm;

    bool write_video;
    string dst_video;
    double dst_video_fps;

    bool make_gray;

    bool resize_src;
    int width, height;

    double scale;
    int nlevels;
    int gr_threshold;

    double hit_threshold;
    bool hit_threshold_auto;

    int win_width;
    int win_stride_width, win_stride_height;
    int block_width;
    int block_stride_width, block_stride_height;
    int cell_width;
    int nbins;

    bool gamma_corr;
};


class App
{
public:
    App(const Args& s);
    void run();

    void handleKey(char key);

    void hogWorkBegin();
    void hogWorkEnd();
    string hogWorkFps() const;

    void workBegin();
    void workEnd();
    string workFps() const;

    string message() const;

    void* thread_display(void *node);

private:
    App operator=(App&);

    unsigned int count = 1;
    Args args;
    bool running;

    bool use_gpu;
    bool make_gray;
    double scale;
    int gr_threshold;
    int nlevels;
    double hit_threshold;
    bool gamma_corr;

    int64 hog_work_begin;
    double hog_work_fps;

    int64 work_begin;
    double work_fps;
};

static void printHelp()
{
    cout << "Histogram of Oriented Gradients descriptor and detector sample.\n"
         << "\nUsage: hog\n"
         << "  (<image>|--video <vide>|--camera <camera_id>) # frames source\n"
         << "  or"
         << "  (--folder <folder_path>) # load images from folder\n"
         << "  [--svm <file> # load svm file"
         << "  [--make_gray <true/false>] # convert image to gray one or not\n"
         << "  [--resize_src <true/false>] # do resize of the source image or not\n"
         << "  [--width <int>] # resized image width\n"
         << "  [--height <int>] # resized image height\n"
         << "  [--hit_threshold <double>] # classifying plane distance threshold (0.0 usually)\n"
         << "  [--scale <double>] # HOG window scale factor\n"
         << "  [--nlevels <int>] # max number of HOG window scales\n"
         << "  [--win_width <int>] # width of the window\n"
         << "  [--win_stride_width <int>] # distance by OX axis between neighbour wins\n"
         << "  [--win_stride_height <int>] # distance by OY axis between neighbour wins\n"
         << "  [--block_width <int>] # width of the block\n"
         << "  [--block_stride_width <int>] # distance by 0X axis between neighbour blocks\n"
         << "  [--block_stride_height <int>] # distance by 0Y axis between neighbour blocks\n"
         << "  [--cell_width <int>] # width of the cell\n"
         << "  [--nbins <int>] # number of bins\n"
         << "  [--gr_threshold <int>] # merging similar rects constant\n"
         << "  [--gamma_correct <int>] # do gamma correction or not\n"
         << "  [--write_video <bool>] # write video or not\n"
         << "  [--dst_video <path>] # output video path\n"
         << "  [--dst_video_fps <double>] # output video fps\n";
    help_showed = true;
}

int main(int argc, char** argv)
{
    try
    {
        Args args;
        if (argc < 2)
        {
            printHelp();
            args.camera_id = 0;
            args.src_is_camera = true;
        }
        else
        {
            args = Args::read(argc, argv);
            if (help_showed)
                return -1;
        }
        App app(args);
        app.run();
    }
    catch (const Exception& e) { return cout << "error: "  << e.what() << endl, 1; }
    catch (const exception& e) { return cout << "error: "  << e.what() << endl, 1; }
    catch(...) { return cout << "unknown exception" << endl, 1; }
    return 0;
}


Args::Args()
{
    src_is_video = false;
    src_is_camera = false;
    src_is_folder = false;
    svm_load = false;
    camera_id = 0;

    write_video = false;
    dst_video_fps = 24.;

    make_gray = true;

    resize_src = false;
    width = 640;
    height = 480;

    scale = 1.05;
    nlevels = 13;
    gr_threshold = 8;
    hit_threshold = 1.4;
    hit_threshold_auto = true;

    win_width = 48;
    win_stride_width = 8;
    win_stride_height = 8;
    block_width = 16;
    block_stride_width = 8;
    block_stride_height = 8;
    cell_width = 8;
    nbins = 9;

    gamma_corr = true;
}


Args Args::read(int argc, char** argv)
{
    Args args;
    for (int i = 1; i < argc; i++)
    {
        if (string(argv[i]) == "--make_gray") args.make_gray = (string(argv[++i]) == "true");
        else if (string(argv[i]) == "--resize_src") args.resize_src = (string(argv[++i]) == "true");
        else if (string(argv[i]) == "--width") args.width = atoi(argv[++i]);
        else if (string(argv[i]) == "--height") args.height = atoi(argv[++i]);
        else if (string(argv[i]) == "--hit_threshold")
        {
            args.hit_threshold = atof(argv[++i]);
            args.hit_threshold_auto = false;
        }
        else if (string(argv[i]) == "--scale") args.scale = atof(argv[++i]);
        else if (string(argv[i]) == "--nlevels") args.nlevels = atoi(argv[++i]);
        else if (string(argv[i]) == "--win_width") args.win_width = atoi(argv[++i]);
        else if (string(argv[i]) == "--win_stride_width") args.win_stride_width = atoi(argv[++i]);
        else if (string(argv[i]) == "--win_stride_height") args.win_stride_height = atoi(argv[++i]);
        else if (string(argv[i]) == "--block_width") args.block_width = atoi(argv[++i]);
        else if (string(argv[i]) == "--block_stride_width") args.block_stride_width = atoi(argv[++i]);
        else if (string(argv[i]) == "--block_stride_height") args.block_stride_height = atoi(argv[++i]);
        else if (string(argv[i]) == "--cell_width") args.cell_width = atoi(argv[++i]);
        else if (string(argv[i]) == "--nbins") args.nbins = atoi(argv[++i]);
        else if (string(argv[i]) == "--gr_threshold") args.gr_threshold = atoi(argv[++i]);
        else if (string(argv[i]) == "--gamma_correct") args.gamma_corr = (string(argv[++i]) == "true");
        else if (string(argv[i]) == "--write_video") args.write_video = (string(argv[++i]) == "true");
        else if (string(argv[i]) == "--dst_video") args.dst_video = argv[++i];
        else if (string(argv[i]) == "--dst_video_fps") args.dst_video_fps = atof(argv[++i]);
        else if (string(argv[i]) == "--help") printHelp();
        else if (string(argv[i]) == "--video") { args.src = argv[++i]; args.src_is_video = true; }
        else if (string(argv[i]) == "--camera") { args.camera_id = atoi(argv[++i]); args.src_is_camera = true; }
        else if (string(argv[i]) == "--folder") { args.src = argv[++i]; args.src_is_folder = true;}
        else if (string(argv[i]) == "--svm") { args.svm = argv[++i]; args.svm_load = true;}
        else if (args.src.empty()) args.src = argv[i];
        else throw runtime_error((string("unknown key: ") + argv[i]));
    }
    return args;
}


App::App(const Args& s)
{
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

    args = s;
    cout << "\nControls:\n"
         << "\tESC - exit\n"
         << "\tm - change mode GPU <-> CPU\n"
         << "\tg - convert image to gray or not\n"
         << "\t1/q - increase/decrease HOG scale\n"
         << "\t2/w - increase/decrease levels count\n"
         << "\t3/e - increase/decrease HOG group threshold\n"
         << "\t4/r - increase/decrease hit threshold\n"
         << endl;

    use_gpu = true;
    make_gray = args.make_gray;
    scale = args.scale;
    gr_threshold = args.gr_threshold;
    nlevels = args.nlevels;

    if (args.hit_threshold_auto)
        args.hit_threshold = args.win_width == 48 ? 1.4 : 0.;
    hit_threshold = args.hit_threshold;

    gamma_corr = args.gamma_corr;

    cout << "Scale: " << scale << endl;
    if (args.resize_src)
        cout << "Resized source: (" << args.width << ", " << args.height << ")\n";
    cout << "Group threshold: " << gr_threshold << endl;
    cout << "Levels number: " << nlevels << endl;
    cout << "Win size: (" << args.win_width << ", " << args.win_width*2 << ")\n";
    cout << "Win stride: (" << args.win_stride_width << ", " << args.win_stride_height << ")\n";
    cout << "Block size: (" << args.block_width << ", " << args.block_width << ")\n";
    cout << "Block stride: (" << args.block_stride_width << ", " << args.block_stride_height << ")\n";
    cout << "Cell size: (" << args.cell_width << ", " << args.cell_width << ")\n";
    cout << "Bins number: " << args.nbins << endl;
    cout << "Hit threshold: " << hit_threshold << endl;
    cout << "Gamma correction: " << gamma_corr << endl;
    cout << endl;
}


void App::run()
{
    running = true;
    cv::VideoWriter video_writer;

    Size win_stride(args.win_stride_width, args.win_stride_height);
    Size win_size(args.win_width, args.win_width * 2);
    Size block_size(args.block_width, args.block_width);
    Size block_stride(args.block_stride_width, args.block_stride_height);
    Size cell_size(args.cell_width, args.cell_width);

    cv::Ptr<cv::cuda::HOG> gpu_hog = cv::cuda::HOG::create(win_size, block_size, block_stride, cell_size, args.nbins);
    cv::HOGDescriptor cpu_hog(win_size, block_size, block_stride, cell_size, args.nbins);

    if(args.svm_load) {
        std::vector<float> svm_model;
        const std::string model_file_name = args.svm;
        FileStorage ifs(model_file_name, FileStorage::READ);
        if (ifs.isOpened()) {
            ifs["svm_detector"] >> svm_model;
        } else {
            const std::string what =
                    "could not load model for hog classifier from file: "
                    + model_file_name;
            throw std::runtime_error(what);
        }

        // check if the variables are initialized
        if (svm_model.empty()) {
            const std::string what =
                    "HoG classifier: svm model could not be loaded from file"
                    + model_file_name;
            throw std::runtime_error(what);
        }

        gpu_hog->setSVMDetector(svm_model);
        cpu_hog.setSVMDetector(svm_model);
    } else {
        // Create HOG descriptors and detectors here
        Mat detector = gpu_hog->getDefaultPeopleDetector();

        gpu_hog->setSVMDetector(detector);
        cpu_hog.setSVMDetector(detector);
    }

    cout << "gpusvmDescriptorSize : " << gpu_hog->getDescriptorSize()
         << endl;
    cout << "cpusvmDescriptorSize : " << cpu_hog.getDescriptorSize()
         << endl;

    // for PGM^RT
    graph_t g;
    node_t  n0, n1, n2, n3, n4, n5, n6, n7, n8, n9;
    edge_t  e0_1, e1_2, e2_3, e3_4, e4_5, e5_6, e6_7, e7_8, e8_9;

    edge_attr_t cv_attr;
    edge_attr_t fast_fifo_attr, fast_mq_attr;
    edge_attr_t fifo_attr, mq_attr;
    // end (for PGM^RT)

    memset(&cv_attr, 0, sizeof(cv_attr));

    printf("sizeof struct params_compute_hists: %d\n", sizeof(struct params_compute_hists));
    cv_attr.nr_produce = sizeof(struct params_compute_hists);
    cv_attr.nr_consume = sizeof(struct params_compute_hists);
    cv_attr.nr_threshold = sizeof(struct params_compute_hists);

    fast_fifo_attr = fast_mq_attr = fifo_attr = mq_attr = cv_attr;

    cv_attr.type = pgm_cv_edge;
    fast_fifo_attr.type = pgm_fast_fifo_edge;
    fast_mq_attr.type = pgm_fast_mq_edge;
    fifo_attr.type = pgm_fifo_edge;
    mq_attr.type = pgm_mq_edge;

    fifo_attr.nr_threshold = nlevels * sizeof(struct params_compute_hists);
    fifo_attr.nr_consume = nlevels * sizeof(struct params_compute_hists);

    fast_mq_attr.mq_maxmsg = nlevels; /* root required for higher values */
    mq_attr.mq_maxmsg = nlevels;

    CheckError(pgm_init("/tmp/graphs", 1));
    CheckError(pgm_init_graph(&g, "demo"));

    CheckError(pgm_init_node(&n0, g, "n0"));
    CheckError(pgm_init_node(&n1, g, "n1"));
    CheckError(pgm_init_node(&n2, g, "n2"));
    CheckError(pgm_init_node(&n3, g, "n3"));
    CheckError(pgm_init_node(&n4, g, "n4"));
    CheckError(pgm_init_node(&n5, g, "n5"));
    CheckError(pgm_init_node(&n6, g, "n6"));
    CheckError(pgm_init_node(&n7, g, "n7"));
    CheckError(pgm_init_node(&n8, g, "n8"));
    CheckError(pgm_init_node(&n9, g, "n9"));

    CheckError(pgm_init_edge(&e0_1, n0, n1, "e0_1", &fast_mq_attr));
    CheckError(pgm_init_edge(&e1_2, n1, n2, "e1_2", &fast_mq_attr));
    CheckError(pgm_init_edge(&e2_3, n2, n3, "e2_3", &fast_mq_attr));
    CheckError(pgm_init_edge(&e3_4, n3, n4, "e3_4", &fast_mq_attr));
    CheckError(pgm_init_edge(&e4_5, n4, n5, "e4_5", &fast_mq_attr));
    CheckError(pgm_init_edge(&e5_6, n5, n6, "e5_6", &fast_mq_attr));
    CheckError(pgm_init_edge(&e6_7, n6, n7, "e6_7", &fifo_attr));
    CheckError(pgm_init_edge(&e7_8, n7, n8, "e7_8", &fast_fifo_attr));
    CheckError(pgm_init_edge(&e8_9, n8, n9, "e8_9", &fast_mq_attr));

    pthread_barrier_init(&gpu_hog->init_barrier, 0, 9);
    init_barrier = &gpu_hog->init_barrier;
    printf("right before creating t1 thread...\n");
    thread t1(&cv::cuda::HOG::thread_detect_multiscale, gpu_hog, &n1);
    thread t2(&cv::cuda::HOG::thread_resize, gpu_hog, &n2);
    thread t3(&cv::cuda::HOG::thread_compute_gradients, gpu_hog, &n3);
    thread t4(&cv::cuda::HOG::thread_compute_hists, gpu_hog, &n4);
    thread t5(&cv::cuda::HOG::thread_normalize_hists, gpu_hog, &n5);
    thread t6(&cv::cuda::HOG::thread_classify_hists, gpu_hog, &n6);
    thread t7(&cv::cuda::HOG::thread_collect_loc, gpu_hog, &n7);
    thread t8(&App::thread_display, this, &n8);
    CheckError(pgm_claim_node(n0));
    CheckError(pgm_claim_node(n9));

    printf("right before barrier in main thread...\n");
    pthread_barrier_wait(&gpu_hog->init_barrier);
    printf("right after barrier in main thread...\n");
    edge_t *out_edge = (edge_t *)calloc(1, sizeof(edge_t));
    CheckError(pgm_get_edges_out(n0, out_edge, 1));
    struct params_compute_hists *out_buf = (struct params_compute_hists *)pgm_get_edge_buf_p(*out_edge);
    if (out_buf == NULL)
        fprintf(stdout, "compute gradients out buffer is NULL\n");
    // end (for PGM^RT)

    while (running)
    {
        Mat frame;
        vector<String> filenames;

        count = 1;

        if (args.src_is_video)
        {
            vc.open(args.src.c_str());
            if (!vc.isOpened())
                throw runtime_error(string("can't open video file: " + args.src));
            vc >> frame;
        }
        else if (args.src_is_folder) {
            String folder = args.src;
            cout << folder << endl;
            glob(folder, filenames);
            frame = imread(filenames[count]);	// 0 --> .gitignore
            if (!frame.data)
                cerr << "Problem loading image from folder!!!" << endl;
        }
        else if (args.src_is_camera)
        {
            vc.open(args.camera_id);
            if (!vc.isOpened())
            {
                stringstream msg;
                msg << "can't open camera: " << args.camera_id;
                throw runtime_error(msg.str());
            }
            vc >> frame;
        }
        else
        {
            frame = imread(args.src);
            if (frame.empty())
                throw runtime_error(string("can't open image file: " + args.src));
        }

        Mat img_aux, img, img_to_show;
        cuda::GpuMat gpu_img;


        // Iterate over all frames
        while (running && !frame.empty())
        {
            workBegin();

            // Change format of the image
            if (make_gray) cvtColor(frame, img_aux, COLOR_BGR2GRAY);
            else if (use_gpu) cvtColor(frame, img_aux, COLOR_BGR2BGRA);
            else frame.copyTo(img_aux);

            // Resize image
            if (args.resize_src) resize(img_aux, img, Size(args.width, args.height));
            else img = img_aux;
            img_to_show = img;

            vector<Rect> found;

            // Perform HOG classification
            hogWorkBegin();
            if (use_gpu)
            {
                gpu_img.upload(img);
                gpu_hog->setNumLevels(nlevels);
                gpu_hog->setHitThreshold(hit_threshold);
                gpu_hog->setWinStride(win_stride);
                gpu_hog->setScaleFactor(scale);
                gpu_hog->setGroupThreshold(gr_threshold);

                out_buf->img_to_show = &img_to_show;
                out_buf->video_writer = video_writer;
                out_buf->filenames = filenames;
                out_buf->frame = &frame;
                out_buf->img = gpu_img;
                out_buf->found_locations = &found;
                out_buf->all_confidences = NULL;

                CheckError(pgm_complete(n0));
                pgm_wait(n9);
            }
            else
            {
                cpu_hog.nlevels = nlevels;
                cpu_hog.detectMultiScale(img, found, hit_threshold, win_stride,
                        Size(0, 0), scale, gr_threshold);
                hogWorkEnd();
                // Draw positive classified windows
                for (size_t i = 0; i < found.size(); i++)
                {
                    Rect r = found[i];
                    rectangle(img_to_show, r.tl(), r.br(), Scalar(0, 255, 0), 3);
                }

                if (use_gpu)
                    putText(img_to_show, "Mode: GPU", Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
                else
                    putText(img_to_show, "Mode: CPU", Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
                putText(img_to_show, "FPS HOG: " + hogWorkFps(), Point(5, 65), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
                putText(img_to_show, "FPS total: " + workFps(), Point(5, 105), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
                imshow("opencv_gpu_hog", img_to_show);

                if (args.src_is_video || args.src_is_camera) vc >> frame;
                if (args.src_is_folder) {
                    count++;
                    if (count < filenames.size()) {
                        frame = imread(filenames[count]);
                    } else {
                        Mat empty;
                        frame = empty;
                    }
                }

                workEnd();

                if (args.write_video)
                {
                    if (!video_writer.isOpened())
                    {
                        video_writer.open(args.dst_video, VideoWriter::fourcc('x','v','i','d'), args.dst_video_fps,
                                img_to_show.size(), true);
                        if (!video_writer.isOpened())
                            throw std::runtime_error("can't create video writer");
                    }

                    if (make_gray) cvtColor(img_to_show, img, COLOR_GRAY2BGR);
                    else cvtColor(img_to_show, img, COLOR_BGRA2BGR);

                    video_writer << img;
                }

                handleKey((char)waitKey(3));
            }

        }
    }
    free(out_edge);
    CheckError(pgm_terminate(n0));
    pthread_barrier_wait(&gpu_hog->init_barrier);
    CheckError(pgm_release_node(n0));
    CheckError(pgm_release_node(n9));
    printf("Joining pthreads...\n");
    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();
    t6.join();
    t7.join();
    t8.join();

    CheckError(pgm_destroy_graph(g));
    CheckError(pgm_destroy());
}

void* App::thread_display(void *_node)
{
    char tabbuf[] = "\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t";
    node_t& node = *((node_t*)_node);
    int ret = 0;

    tabbuf[node.node] = '\0';

    CheckError(pgm_claim_node(node));

    edge_t *in_edge = (edge_t *)calloc(1, sizeof(edge_t));
    CheckError(pgm_get_edges_in(node, in_edge, 1));
    struct params_compute_hists *in_buf = (struct params_compute_hists *)pgm_get_edge_buf_c(*in_edge);
    if (in_buf == NULL)
        fprintf(stderr, "compute gradients in buffer is NULL\n");
    edge_t *out_edge = (edge_t *)calloc(1, sizeof(edge_t));
    CheckError(pgm_get_edges_out(node, out_edge, 1));
    struct params_compute_hists *out_buf = (struct params_compute_hists *)pgm_get_edge_buf_p(*out_edge);
    if (out_buf == NULL)
        fprintf(stderr, "compute gradients out buffer is NULL\n");

    pthread_barrier_wait(init_barrier);

    if(!hog_sample_errors)
    {
        do {
            ret = pgm_wait(node);

            if(ret != PGM_TERMINATE)
            {
                CheckError(ret);
                hogWorkEnd();
#if DEBUG
                fprintf(stdout, "%s%d fires\n", tabbuf, node.node);
#endif
                // Draw positive classified windows
                for (size_t i = 0; i < in_buf->found_locations->size(); i++)
                {
                    Rect r = (*in_buf->found_locations)[i];
                    rectangle(*in_buf->img_to_show, r.tl(), r.br(), Scalar(0, 255, 0), 3);
                }

                if (use_gpu)
                    putText(*in_buf->img_to_show, "Mode: GPU", Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
                else
                    putText(*in_buf->img_to_show, "Mode: CPU", Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
                putText(*in_buf->img_to_show, "FPS HOG: " + hogWorkFps(), Point(5, 65), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
                putText(*in_buf->img_to_show, "FPS total: " + workFps(), Point(5, 105), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
                imshow("opencv_gpu_hog", *in_buf->img_to_show);

                if (args.src_is_video || args.src_is_camera) vc >> *in_buf->frame;
                if (args.src_is_folder) {
                    count++;
                    if (count < in_buf->filenames.size()) {
                        *in_buf->frame = imread(in_buf->filenames[count]);
                    } else {
                        in_buf->frame = NULL;
                    }
                }

                workEnd();

                if (args.write_video)
                {
                    Mat img;
                    if (!in_buf->video_writer.isOpened())
                    {
                        in_buf->video_writer.open(args.dst_video, VideoWriter::fourcc('x','v','i','d'), args.dst_video_fps,
                                in_buf->img_to_show->size(), true);
                        if (!in_buf->video_writer.isOpened())
                            throw std::runtime_error("can't create video writer");
                    }

                    if (make_gray) cvtColor(*in_buf->img_to_show, img, COLOR_GRAY2BGR);
                    else cvtColor(*in_buf->img_to_show, img, COLOR_BGRA2BGR);

                    in_buf->video_writer << img;
                }

                CheckError(pgm_swap_edge_bufs(in_buf, out_buf));
                struct params_compute_hists *temp = in_buf;
                in_buf = out_buf;
                out_buf = temp;
                CheckError(pgm_complete(node));
            }
            else
            {
                fprintf(stdout, "%s- %d terminates\n", tabbuf, node.node);
            }

            handleKey((char)waitKey(3));
        } while(ret != PGM_TERMINATE);
    }

    pthread_barrier_wait(init_barrier);

    CheckError(pgm_release_node(node));

    free(in_edge);
    pthread_exit(0);

}


void App::handleKey(char key)
{
    switch (key)
    {
    case 27:
        running = false;
        break;
    case 'm':
    case 'M':
        use_gpu = !use_gpu;
        cout << "Switched to " << (use_gpu ? "CUDA" : "CPU") << " mode\n";
        break;
    case 'g':
    case 'G':
        make_gray = !make_gray;
        cout << "Convert image to gray: " << (make_gray ? "YES" : "NO") << endl;
        break;
    case '1':
        scale *= 1.05;
        cout << "Scale: " << scale << endl;
        break;
    case 'q':
    case 'Q':
        scale /= 1.05;
        cout << "Scale: " << scale << endl;
        break;
    case '2':
        nlevels++;
        cout << "Levels number: " << nlevels << endl;
        break;
    case 'w':
    case 'W':
        nlevels = max(nlevels - 1, 1);
        cout << "Levels number: " << nlevels << endl;
        break;
    case '3':
        gr_threshold++;
        cout << "Group threshold: " << gr_threshold << endl;
        break;
    case 'e':
    case 'E':
        gr_threshold = max(0, gr_threshold - 1);
        cout << "Group threshold: " << gr_threshold << endl;
        break;
    case '4':
        hit_threshold+=0.25;
        cout << "Hit threshold: " << hit_threshold << endl;
        break;
    case 'r':
    case 'R':
        hit_threshold = max(0.0, hit_threshold - 0.25);
        cout << "Hit threshold: " << hit_threshold << endl;
        break;
    case 'c':
    case 'C':
        gamma_corr = !gamma_corr;
        cout << "Gamma correction: " << gamma_corr << endl;
        break;
    }
}


inline void App::hogWorkBegin() { hog_work_begin = getTickCount(); }

inline void App::hogWorkEnd()
{
    int64 delta = getTickCount() - hog_work_begin;
    double freq = getTickFrequency();
    hog_work_fps = freq / delta;
}

inline string App::hogWorkFps() const
{
    stringstream ss;
    ss << hog_work_fps;
    return ss.str();
}


inline void App::workBegin() { work_begin = getTickCount(); }

inline void App::workEnd()
{
    int64 delta = getTickCount() - work_begin;
    double freq = getTickFrequency();
    work_fps = freq / delta;
}

inline string App::workFps() const
{
    stringstream ss;
    ss << work_fps;
    return ss.str();
}
