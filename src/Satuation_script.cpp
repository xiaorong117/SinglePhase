#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <ctime>
#include <vector>
#include <cstring>
#include <cstdio>

int main(int argc, char const *argv[])
{
    for (size_t i = 0; i < 10; i++)
    {
        double num{0.04 * i};
        double voxel_clay{0.404 * (1 - num)};
        double voxel_OM_HP{0};
        double voxel_OM_LP{0.025 + 0.404 * num};
        double porosity_clay{0};
        double porosity_OM_HP{0};
        double porosity_OM_LP{0};
        double e_macro_clay{8.3 / 100};
        double e_macro_OM_HP{24.3 / 100};
        double e_macro_OM_LP{8.3 / 100};
        double e_micro{double(10) / double(100)};

        double local_Sw{0};
        double local_SW_OM{0};

        porosity_OM_HP = (voxel_OM_HP * e_macro_OM_HP + voxel_OM_HP * (1 - e_macro_OM_HP) * e_micro);
        porosity_OM_LP = (voxel_OM_LP * e_macro_OM_LP + voxel_OM_LP * (1 - e_macro_OM_LP) * e_micro);
        porosity_clay = (voxel_clay * e_macro_clay + voxel_clay * (1 - e_macro_clay) * e_micro);

        // std::ofstream outfile("2B_clay_Global_Sw.txt");
        // for (size_t i = 0; i < 101; i++)
        // {
        //     local_Sw = i * 0.01;
        //     outfile << porosity_clay * local_Sw/ (porosity_clay + porosity_OM_HP + porosity_OM_LP) << " " << local_Sw << std::endl;
        // }
        // outfile << "macro porosity of clay = " << e_macro_clay / (e_macro_clay + (1 - e_macro_clay) * e_micro) << " " << "micro porosity of clay = " << (1 - e_macro_clay) * e_micro / (e_macro_clay + (1 - e_macro_clay) * e_micro) << std::endl;
        // outfile << "macro porosity of OM-HP = " << e_macro_OM_HP / (e_macro_OM_HP + (1 - e_macro_OM_HP) * e_micro) << " " << "micro porosity of OM-HP = " << (1 - e_macro_OM_HP) * e_micro / (e_macro_OM_HP + (1 - e_macro_OM_HP) * e_micro) << std::endl;
        // outfile.close();
        std::string filename1(std::to_string(voxel_clay));
        std::string filename2(std::to_string(voxel_OM_LP));
        std::string filename3(std::to_string(i));
        std::ofstream outfile1("35jx_Global_Sw_model1_" + filename1 + "_" + filename2 + "_" + filename3 + ".txt");
        for (size_t i = 0; i < 11; i++)
        {
            local_Sw = i * 0.1;
            local_SW_OM = local_Sw / 2;
            double Sw_clay_global = porosity_clay * local_Sw / (porosity_clay + porosity_OM_HP + porosity_OM_LP);
            double Sw_OM_global = porosity_OM_HP * local_SW_OM / (porosity_clay + porosity_OM_HP + porosity_OM_LP);
            double Sw_OMLP_global = porosity_OM_LP * local_SW_OM / (porosity_clay + porosity_OM_HP + porosity_OM_LP);
            outfile1 << Sw_clay_global + Sw_OM_global + Sw_OMLP_global << std::endl; // << " " << local_Sw 
        }
        // outfile1 << "macro porosity of clay = " << e_macro_clay / (e_macro_clay + (1 - e_macro_clay) * e_micro) << " " << "micro porosity of clay = " << (1 - e_macro_clay) * e_micro / (e_macro_clay + (1 - e_macro_clay) * e_micro) << std::endl;
        // outfile1 << "macro porosity of OM-HP = " << e_macro_OM_HP / (e_macro_OM_HP + (1 - e_macro_OM_HP) * e_micro) << " " << "micro porosity of OM-HP = " << (1 - e_macro_OM_HP) * e_micro / (e_macro_OM_HP + (1 - e_macro_OM_HP) * e_micro) << std::endl;
        outfile1.close();

        std::ofstream outfile("35jx_Global_Sw_model2_" + filename1 + "_" + filename2 + "_" + filename3 + ".txt");
        for (size_t i = 0; i < 6; i++)
        {
            local_Sw = 1;
            local_SW_OM = 0.5 + i * 0.1;
            double Sw_clay_global = porosity_clay * local_Sw / (porosity_clay + porosity_OM_HP + porosity_OM_LP);
            double Sw_OM_global = porosity_OM_HP * local_SW_OM / (porosity_clay + porosity_OM_HP + porosity_OM_LP);
            double Sw_OMLP_global = porosity_OM_LP * local_SW_OM / (porosity_clay + porosity_OM_HP + porosity_OM_LP);
            outfile << Sw_clay_global + Sw_OM_global + Sw_OMLP_global << std::endl; //<< " " << local_SW_OM 
        }
        // outfile << "macro porosity of clay = " << e_macro_clay / (e_macro_clay + (1 - e_macro_clay) * e_micro) << " " << "micro porosity of clay = " << (1 - e_macro_clay) * e_micro / (e_macro_clay + (1 - e_macro_clay) * e_micro) << std::endl;
        // outfile << "macro porosity of OM-HP = " << e_macro_OM_HP / (e_macro_OM_HP + (1 - e_macro_OM_HP) * e_micro) << " " << "micro porosity of OM-HP = " << (1 - e_macro_OM_HP) * e_micro / (e_macro_OM_HP + (1 - e_macro_OM_HP) * e_micro) << std::endl;
        outfile.close();
    }
    // double KK = 0;
    // std::ofstream outfile("Kclay_Sw.txt");
    // double Sw_max_clay{1};
    // double a_clay = (1.2e-21 - KK) / pow(Sw_max_clay, 6);
    // for (size_t i = 0; i < 41; i++)
    // {
    //     double Sw = 0.01 * i;
    //     double K_Clay{a_clay * pow(Sw_max_clay - Sw, 6) + KK};
    //     outfile << Sw << "\t" << K_Clay/1e-21 << std::endl;
    // }
    // outfile.close();

    // std::ofstream outfile2("KOMA_Sw.txt");
    // double Sw_max_omA{1};
    // double a_omA = (743e-21 - KK) / pow(Sw_max_omA, 5.5);
    // for (size_t i = 0; i < 81; i++)
    // {
    //     double Sw = 0.01 * i;
    //     double K_OM{a_omA * pow(Sw_max_omA - Sw, 5.5) + KK};
    //     outfile2 <<Sw  << "\t" <<  K_OM/1e-21<< std::endl;
    // }
    // outfile2.close();

    // std::ofstream outfile3("KOMB_Sw.txt");
    // double Sw_max_omB{1};
    // double a_omB = (2e-21 - KK) / pow(Sw_max_omB, 3);
    // for (size_t i = 0; i < 61; i++)
    // {
    //     double Sw = 0.01 * i;
    //     double K_OM{a_omB * pow(Sw_max_omB - Sw, 3) + KK};
    //     outfile3 << Sw << "\t" << K_OM/1e-21 << std::endl;
    // }
    // outfile3.close();

    // std::ofstream outfile4("KOMC_Sw.txt");
    // double Sw_max_omC{1};
    // double a_omC = (1.5e-21 - KK) / pow(Sw_max_omC, 3);
    // for (size_t i = 0; i < 61; i++)
    // {
    //     double Sw = 0.01 * i;
    //     double K_OM{a_omC * pow(Sw_max_omC - Sw, 3) + KK};
    //     outfile4 <<Sw   << "\t" << K_OM/1e-21<< std::endl;
    // }
    // outfile4.close();

    return 0;
}