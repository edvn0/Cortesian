//
// Created by Edwin Carlsson on 2021-06-03.
//

#ifndef CORTESIAN_METABASE_H
#define CORTESIAN_METABASE_H

#include <string>
#include <unordered_map>
#include <vector>

/**
 * Provides a simple "Parent" to all objects for serialization/deserialization.
 * This might be the absolute worst way to handle this.
 */
class MetaBase {
protected:
  std::unordered_map<std::string, std::string> m_meta_data;

  virtual void operator()(std::string &key, std::string &value) {
    m_meta_data[key] = value;
  };

  virtual void operator()(std::string &&key, std::string &&value) {
    m_meta_data[key] = value;
  };

  virtual void operator()(std::string &key, std::string &&value) {
    m_meta_data[key] = value;
  };

  virtual void operator()(std::string &&key, std::string &value) {
    m_meta_data[key] = value;
  };

public:
  virtual std::string operator[](const std::string &key) {
    return m_meta_data[key];
  };

  /**
   * Return the tuples from the meta data of subclassing entities.
   * @return vector of k,v pairs of meta data.
   */
  [[nodiscard]] const std::vector<std::tuple<std::string, std::string>>
  key_value_pairs() const {
    std::vector<std::tuple<std::string, std::string>> key_value_pairs;
    key_value_pairs.reserve(m_meta_data.size());
    for (auto &tuple : m_meta_data) {
      key_value_pairs.emplace_back(tuple);
    }

    return key_value_pairs;
  }
};

#endif // CORTESIAN_METABASE_H
