// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/ngraph.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "functional_test_utils/skip_tests_config.hpp"
#include "low_precision/layer_transformation.hpp"


#include "low_precision_transformations/clamp_transformation.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {
typedef std::tuple<ngraph::element::Type,
                   ngraph::PartialShape,
                   std::string,
                   int> FakeParams;

class GatherSnippetsTransformation : public testing::WithParamInterface<FakeParams>,
                                     public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<FakeParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions

namespace LayerTestsDefinitions {

std::string GatherSnippetsTransformation::getTestCaseName(const testing::TestParamInfo<FakeParams>& obj) {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShape;
    std::string targetDevice;
    int param;
    std::tie(netPrecision, inputShape, targetDevice, param) = obj.param;

    std::ostringstream result;
    result << "Snippets_Gather_Test";
    return result.str();
}

void GatherSnippetsTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShape;
    int param;
    std::tie(netPrecision, inputShape, targetDevice, param) = GetParam();

    const InferenceEngine::SizeVector _kernel = {3, 3};
    const InferenceEngine::SizeVector _stride = {1, 1};
    const InferenceEngine::SizeVector _dilation = {1, 1};
    const std::vector<ptrdiff_t> _padBegin = {0, 0};
    const std::vector<ptrdiff_t> _padEnd = {0, 0};
    const size_t _convOutChannels = 64;

    auto params = ngraph::builder::makeParams(ngraph::element::f32, {inputShape.to_shape()});

    auto conv = ngraph::builder::makeConvolution(params[0],
                                                 ngraph::element::f32,
                                                 _kernel,
                                                 _stride,
                                                 _padBegin,
                                                 _padEnd,
                                                 _dilation,
                                                 ngraph::op::PadType::EXPLICIT,
                                                 _convOutChannels);

    auto sum_const = ngraph::opset1::Constant::create(ngraph::element::f32, {1, 64, 1, 1}, {2.f});
    auto sum = std::make_shared<ngraph::opset1::Add>(conv, sum_const);

    auto sum_const_2 = ngraph::opset1::Constant::create(ngraph::element::f32, {1, 64, 1, 1}, {3.f});
    auto sum_2 = std::make_shared<ngraph::opset1::Add>(conv, sum_const_2);

    auto indices = ngraph::opset1::Constant::create(ngraph::element::i32, {2}, {2, 3});
    auto axis = ngraph::opset1::Constant::create(ngraph::element::i32, {}, {1});
    auto gather = std::make_shared<ngraph::opset8::Gather>(sum, indices, axis);

    auto indices_2 = ngraph::opset1::Constant::create(ngraph::element::i32, {2}, {2, 3});
    auto axis_2 = ngraph::opset1::Constant::create(ngraph::element::i32, {}, {1});
    auto gather_2 = std::make_shared<ngraph::opset8::Gather>(sum_2, indices_2, axis_2);

    auto mul = std::make_shared<ngraph::opset1::Multiply>(gather, gather_2);

    function = std::make_shared<ngraph::Function>(ngraph::OutputVector{mul}, "SnippetsTest");
}

TEST_P(GatherSnippetsTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run();
};

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;
INSTANTIATE_TEST_SUITE_P(Custom_test,
                         GatherSnippetsTransformation,
                         ::testing::Combine(::testing::Values(ngraph::element::f32),
                                            ::testing::Values(ngraph::PartialShape({1, 3, 16, 16})),
                                            ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                            ::testing::Values(1)),
                         GatherSnippetsTransformation::getTestCaseName);
