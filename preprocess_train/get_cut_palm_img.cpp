#include "get_cut_img.hpp"

int CountLines(char *filename)
{
    std::ifstream ReadFile;
    int n = 0;
    std::string tmp;
    ReadFile.open(filename, std::ios::in); // read only

    while (getline(ReadFile, tmp, '\n')) {
        n++;
    }
    ReadFile.close();
    return n;
}


int get_img(char *ImgPath, char *ResultName_, int Dst_img_width,
            int Dst_img_height, int ImgBeginX, int ImgBeginY)
{
    cv::Mat Img = cv::imread(ImgPath);
    if (!Img.data) { cout << "can't open img" << endl; return -1; }

    cv::Mat SubImg = Mat::zeros(Dst_img_height, Dst_img_width, CV_8UC3);
    for (int i = 0; i<Dst_img_height; i++) {
        for (int j = 0; j<Dst_img_width; j++) {
            SubImg.at<Vec3b>(i, j) = Img.at<Vec3b>
                    (ImgBeginY + i, ImgBeginX + j);
        }
    }

    std::string ResultName = ResultName_;
    cv::imwrite(ResultName, SubImg);

    return 0;

}


int cut_img(char *palm_file, char *palmxyfile, int N_small_img, int Size_small_img,
            char *palm_box, char *img_name_, int ImgBeginX, int ImgBeginY )
{
    srand((unsigned)time(NULL));
    Mat youngpalm   = imread(palm_file);
    Mat palm        = youngpalm.clone();
    int nline       = CountLines(palmxyfile);
    int nwidth      = youngpalm.cols;
    int nheight     = youngpalm.rows;

    ofstream palmboxfile(palm_box, ios::app);
    for (int i = 0; i < N_small_img; i++)
    {
        int X_Top_Left_position = (int)rand() % (nwidth  - Size_small_img);
        int Y_Top_Left_position = (int)rand() % (nheight - Size_small_img);
        int X_Bottom_Right_position = X_Top_Left_position + Size_small_img;
        int Y_Bottom_Right_position = Y_Top_Left_position + Size_small_img;
        string filename = "0000.jpg";
        std::string img_name = img_name_; // the name of each training sample image

        if (i < 10) {
            filename[0] = '0';        
            filename[1] = '0';
            filename[2] = '0';
            filename[3] = i % 10 + '0';
        }
        else if (i < 100) {
            filename[0] = '0';
            filename[1] = '0';
            filename[2] = i / 10 + '0';
            filename[3] = i % 10 + '0';
        }
        else if (i < 1000) {
            filename[0] = '0';
            filename[1] = i / 100 + '0';
            filename[2] = (i % 100) / 10 + '0';
            filename[3] = i % 100 % 10 + '0';
        }
        else {
            filename[0] = i / 1000 + '0';
            filename[1] = (i / 100) % 10 + '0';
            filename[2] = (i % 100) / 10 + '0';
            filename[3] = (i % 100) % 10 + '0';
        }
        img_name += filename;
        ifstream xyFile(palmxyfile);

        double **palmxyarray = new double *[nline];
        double **BOX_xyarray = new double *[nline];

        int index = 0;
        for (int j = 0; j < nline; j++) {

            palmxyarray[j] = new double[6];
            BOX_xyarray[j] = new double[5];
            xyFile >> palmxyarray[j][4];
            xyFile >> palmxyarray[j][0];
            xyFile >> palmxyarray[j][1];
            xyFile >> palmxyarray[j][2];
            xyFile >> palmxyarray[j][3];
            xyFile >> palmxyarray[j][5];
            int box_left_x = palmxyarray[j][0]  - ImgBeginX;
            int box_left_y = palmxyarray[j][1]  - ImgBeginY;
            int box_right_x = palmxyarray[j][2] - ImgBeginX;
            int box_right_y = palmxyarray[j][3] - ImgBeginY;
            if (box_left_x > X_Top_Left_position && box_left_y >
                Y_Top_Left_position && box_right_x < X_Bottom_Right_position &&
                box_right_y < Y_Bottom_Right_position) {

                BOX_xyarray[index][0] = box_left_x  - X_Top_Left_position;
                BOX_xyarray[index][1] = box_left_y  - Y_Top_Left_position;
                BOX_xyarray[index][2] = box_right_x - X_Top_Left_position;
                BOX_xyarray[index][3] = box_right_y - Y_Top_Left_position;
                BOX_xyarray[index][4] = palmxyarray[j][5];
                index++;

            }
        }
        if (index == 0) continue;

        cv::Mat SubImg = Mat::zeros(Size_small_img, Size_small_img, CV_8UC3);
        for (int ii = 0; ii<Size_small_img; ii++) {
            for (int jj = 0; jj<Size_small_img; jj++) {
                SubImg.at<Vec3b>(ii, jj) = youngpalm.at<Vec3b>
                        (Y_Top_Left_position + ii, X_Top_Left_position + jj);
            }
        }
        cv::imwrite(img_name, SubImg);

        palmboxfile << img_name << " ";
        palmboxfile << index << " ";
        for (int k = 0; k < index; k++) {
            palmboxfile << BOX_xyarray[k][0] << " ";
            palmboxfile << BOX_xyarray[k][1] << " ";
            palmboxfile << BOX_xyarray[k][2] << " ";
            palmboxfile << BOX_xyarray[k][3] << " ";
            palmboxfile << (int)BOX_xyarray[k][4] << " ";
        }
        palmboxfile << endl;
    }

    return 0;
}



