// Physical_property_cal.hpp

#ifndef PHYSICAL_PROPERTY_CAL_HPP
#define PHYSICAL_PROPERTY_CAL_HPP

class Physical_property_cal {
 private:
  /* data */
 public:
  Physical_property();
  double compre(double pressure);        // 压缩系数
  double visco(double pressure, double z, double T);
  void Function_DS(double pressure);
  double Function_Slip(double knusen);
  double Function_Slip_clay(double knusen);
  ~Physical_property();
};

#endif        // PHYSICAL_PROPERTY_CAL_HPP