//
// Created by Edwin Carlsson on 2021-06-03.
//

#ifndef CORTESIAN_METABASE_H
#define CORTESIAN_METABASE_H

#include <string>
#include <unordered_map>
#include <vector>

class MetaBase {
protected:
  std::unordered_map<std::string, std::string> m_meta_data;

  virtual void operator()(std::string &key, std::string &value) {
    m_meta_data[key] = value;
  };

  virtual void operator()(std::string &&key, std::string &&value) {
    m_meta_data[key] = value;
  };

public:
  virtual std::string operator[](const std::string &key) {
    return m_meta_data[key];
  };

  [[nodiscard]] std::vector<std::string> key_value_pairs() const {
    std::vector<std::string> key_value_pairs;
    key_value_pairs.reserve(m_meta_data.size());
    for (auto &[key, value] : m_meta_data) {
      std::string key_value;
      key_value.append(key);
      key_value.append(",");
      key_value.append(value);
      key_value_pairs.emplace_back(key_value);
    }

    return key_value_pairs;
  }
};

#endif // CORTESIAN_METABASE_H
