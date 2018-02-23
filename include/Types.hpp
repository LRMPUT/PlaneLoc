/*
    Copyright (c) 2017 Mobile Robots Laboratory at Poznan University of Technology:
    -Jan Wietrzykowski name.surname [at] put.poznan.pl

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#ifndef INCLUDE_TYPES_HPP_
#define INCLUDE_TYPES_HPP_

#include <Eigen/Eigen>
#include <Eigen/StdVector>
#include <Eigen/StdList>

typedef Eigen::Matrix<double, 6, 1, Eigen::ColMajor> Vector6d;
typedef Eigen::Matrix<double, 7, 1, Eigen::ColMajor> Vector7d;

class ObjInstance;
class PlaneSeg;
class LineSeg;

typedef std::vector<ObjInstance, Eigen::aligned_allocator<ObjInstance> > vectorObjInstance;
typedef std::vector<PlaneSeg, Eigen::aligned_allocator<PlaneSeg> > vectorPlaneSeg;
typedef std::vector<LineSeg, Eigen::aligned_allocator<LineSeg> > vectorLineSeg;
typedef std::vector<Vector7d, Eigen::aligned_allocator<Vector7d> > vectorVector7d;
typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > vectorVector2d;
typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vectorVector3d;
typedef std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > vectorVector4d;

typedef std::list<ObjInstance, Eigen::aligned_allocator<ObjInstance> > listObjInstance;

#endif /* INCLUDE_TYPES_HPP_ */
