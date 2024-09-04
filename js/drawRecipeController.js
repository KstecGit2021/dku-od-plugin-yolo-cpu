// 'detectionRecipe.draw'라는 AngularJS 모듈을 생성
var app = angular.module('detectionRecipe.draw', []);

// 'drawRecipeController'라는 이름의 컨트롤러를 정의하고, $scope를 주입
app.controller('drawRecipeController', function($scope) {

    // 특정 변수(varName)가 정의되지 않았을 경우 초기값(initValue)으로 설정하는 함수
    var initVariable = function(varName, initValue) {
        if ($scope.config[varName] == undefined) {
            $scope.config[varName] = initValue; // 변수의 초기값 설정
        }
    };

    // 모델의 신뢰도(confidence) 정보를 가져오는 함수 정의
    var retrieveInfoRetrain = function() {
        // Python 메소드를 호출하여 신뢰도 정보를 가져옴
        $scope.callPythonDo({method: "get-confidence"}).then(function(data) {
            // 신뢰도 정보가 있는지 여부를 스코프에 설정
            $scope.hasConfidence = data["has_confidence"];
            $scope.finishedLoading = true; // 로딩 완료 상태로 설정
        }, function(data) {
            $scope.finishedLoading = true; // 로딩 완료 상태로 설정
        });
    };

    // 여러 변수를 초기화하는 함수 정의
    var initVariables = function() {
        initVariable("draw_label", true); // 레이블 그리기 옵션 초기화
        initVariable("draw_confidence", false); // 신뢰도 표시 옵션 초기화
    };

    // 컨트롤러 초기화 함수 정의
    var init = function() {
        initVariables(); // 변수들 초기화
        retrieveInfoRetrain(); // 신뢰도 정보 가져오기
    };

    // 컨트롤러 초기화 함수 호출
    init();
});
