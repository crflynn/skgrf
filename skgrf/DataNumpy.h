#ifndef DATANUMPY_H_
#define DATANUMPY_H_

#include "globals.h"
#include "Data.h"

namespace grf {

class DataNumpy final: public Data {
public:
  DataNumpy();

  DataNumpy(double* data, size_t num_rows, size_t num_cols) {
    std::vector<double> datavec(data, data + num_cols * num_rows);
    this->data = datavec;
    this->num_rows = num_rows;
    this->num_cols = num_cols;
  };

  ~DataNumpy() = default;

  double get(size_t row, size_t col) const {
    return data[col * num_rows + row];
  };

  void reserve_memory() {
    data.resize(num_cols * num_rows);
  };

  void set(size_t col, size_t row, double value, bool& error) {
    data[col * num_rows + row] = value;
  };

private:
  std::vector<double> data;

  DISALLOW_COPY_AND_ASSIGN(DataNumpy);
};

} // namespace ranger

#endif
