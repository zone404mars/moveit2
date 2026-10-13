/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2026, Z404
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/** @file
 * @brief Tests for the STOMP update filters: position bounds must clip positions, not updates.
 */

#include <gtest/gtest.h>
#include <moveit/utils/robot_model_test_utils.hpp>
#include <stomp_moveit/filter_functions.hpp>

namespace
{
constexpr size_t TIMESTEPS = 40;
constexpr double LOWER = -2.95;  // [rad] a joint range that excludes 0
constexpr double UPPER = -0.3;   // [rad]
constexpr double TOLERANCE = 1e-9;

// A single revolute joint bounded to [LOWER, UPPER]
moveit::core::RobotModelPtr makeModel()
{
  moveit::core::RobotModelBuilder builder("one_joint", "base");
  builder.addChain("base->link", "revolute");
  builder.addGroupChain("base", "link", "arm");
  moveit::core::RobotModelPtr model = builder.build();
  moveit::core::VariableBounds bounds;
  bounds.position_bounded_ = true;
  bounds.min_position_ = LOWER;
  bounds.max_position_ = UPPER;
  moveit::core::JointModel* joint = model->getJointModel("base-link-joint");
  joint->setVariableBounds(joint->getName(), bounds);
  return model;
}
}  // namespace

TEST(FilterFunctionsTest, enforcePositionBoundsKeepsValidUpdates)
{
  // GIVEN a joint at -1.57 rad on all waypoints, well inside its bounds, and an update of +0.1 rad
  const auto model = makeModel();
  const auto* group = model->getJointModelGroup("arm");
  const Eigen::MatrixXd parameters = Eigen::MatrixXd::Constant(1, TIMESTEPS, -1.57);
  Eigen::MatrixXd updates = Eigen::MatrixXd::Constant(1, TIMESTEPS, 0.1);

  // WHEN the bounds filter is applied to the updates, as STOMP does after each iteration
  stomp_moveit::filters::enforcePositionBounds(group)(parameters, updates);

  // THEN the update is unchanged, since the resulting position -1.47 rad is valid
  EXPECT_TRUE(updates.isApprox(Eigen::MatrixXd::Constant(1, TIMESTEPS, 0.1), TOLERANCE)) << updates;
}

TEST(FilterFunctionsTest, enforcePositionBoundsClipsResultingPositions)
{
  // GIVEN a joint at -0.35 rad and an update of +0.2 rad that would leave the bounds
  const auto model = makeModel();
  const auto* group = model->getJointModelGroup("arm");
  const Eigen::MatrixXd parameters = Eigen::MatrixXd::Constant(1, TIMESTEPS, -0.35);
  Eigen::MatrixXd updates = Eigen::MatrixXd::Constant(1, TIMESTEPS, 0.2);

  stomp_moveit::filters::enforcePositionBounds(group)(parameters, updates);

  // THEN the update is reduced so that the position lands on the upper bound
  const Eigen::MatrixXd positions = parameters + updates;
  EXPECT_TRUE(positions.isApprox(Eigen::MatrixXd::Constant(1, TIMESTEPS, UPPER), TOLERANCE)) << positions;
}

TEST(FilterFunctionsTest, plannerFilterChainDoesNotShiftValidTrajectory)
{
  // GIVEN the filter chain used by the STOMP planning context, a valid trajectory and zero updates
  const auto model = makeModel();
  const auto* group = model->getJointModelGroup("arm");
  const auto filter_fn = stomp_moveit::filters::chain(
      { stomp_moveit::filters::simpleSmoothingMatrix(TIMESTEPS), stomp_moveit::filters::enforcePositionBounds(group) });
  const Eigen::MatrixXd parameters = Eigen::MatrixXd::Constant(1, TIMESTEPS, -1.57);
  Eigen::MatrixXd updates = Eigen::MatrixXd::Zero(1, TIMESTEPS);

  // WHEN the updates are filtered
  filter_fn(parameters, updates);

  // THEN the trajectory does not move, start and goal waypoints included
  EXPECT_NEAR(updates(0, 0), 0.0, TOLERANCE);
  EXPECT_NEAR(updates(0, TIMESTEPS - 1), 0.0, TOLERANCE);
  EXPECT_TRUE(updates.isZero(TOLERANCE)) << updates;
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
