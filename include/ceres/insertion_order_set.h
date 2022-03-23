#ifndef CERES_INSERTION_ORDER_SET_H_
#define CERES_INSERTION_ORDER_SET_H_

#include <algorithm>
#include <vector>
#include <unordered_set>

namespace ceres {

template <typename T>
class InsertionOrderSet
{
 public:
  using iterator = typename std::vector<T>::iterator;
  using const_iterator = typename std::vector<T>::const_iterator;

  InsertionOrderSet() = default;

  InsertionOrderSet(std::initializer_list<T> l)
  {
    for (const T& element : l)
      insert(element);
  }

  std::size_t size() const
  {
    return vec_.size();
  }

  bool empty() const
  {
    return vec_.empty();
  }

  void insert(const T& element)
  {
    if (!unique_vals_.count(element))
    {
      vec_.push_back(element);
      unique_vals_.insert(element);
    }
  }

  template <class InputIt>
  void insert(InputIt first, InputIt last)
  {
    for (InputIt it = first; it < last; it++)
      insert(*it);
  }

  bool operator==(const InsertionOrderSet<T>& other) const
  {
    return vec_ == other.vec_;
  }

  void erase(const T& element)
  {
    if (unique_vals_.count(element))
    {
      iterator it = std::find(vec_.begin(), vec_.end(), element);
      vec_.erase(it);
      unique_vals_.erase(element);
    }
  }

  int count(const T& element) const
  {
    return unique_vals_.count(element);
  }

  iterator begin()
  {
    return vec_.begin();
  }

  const_iterator begin() const
  {
    return vec_.begin();
  }

  iterator end()
  {
    return vec_.end();
  }

  const_iterator end() const
  {
    return vec_.end();
  }

  iterator find(const T& val)
  {
    return std::find(vec_.begin(), vec_.end(), val);
  }

  const_iterator find(const T& val) const
  {
    return std::find(vec_.begin(), vec_.end(), val);
  }

 protected:
  std::vector<T> vec_;
  std::unordered_set<T> unique_vals_;
};
}  // namespace ceres

#endif  // CERES_INSERTION_ORDER_SET_H_
