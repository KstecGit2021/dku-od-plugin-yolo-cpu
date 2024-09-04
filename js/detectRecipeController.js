// AngularJS 모듈을 생성하고 'detectionRecipe.detect'라는 이름으로 정의
var app = angular.module('detectionRecipe.detect', []);

// 'detectRecipeController'라는 컨트롤러를 정의하고, $scope를 주입
app.controller('detectRecipeController', function($scope) {

    // GPU 사용 가능 여부를 확인하는 함수 정의
    var retrieveCanUseGPU = function() {
        // Python 메소드를 호출하여 GPU 정보를 가져옴
        $scope.callPythonDo({method: "get-gpu-info"}).then(function(data) {
            // GPU를 사용할 수 있는지 여부를 스코프에 설정
            $scope.canUseGPU = data["can_use_gpu"];
            $scope.finishedLoading = true; // 로딩 완료 상태로 설정
        }, function(data) {
            // GPU 사용 불가 시 처리
            $scope.canUseGPU = false;
            $scope.finishedLoading = true; // 로딩 완료 상태로 설정
        });
    };

    // 변수 초기화 함수 정의 (변수가 정의되지 않은 경우에만 초기값 설정)
    var initVariable = function(varName, initValue) {
        if ($scope.config[varName] == undefined) {
            $scope.config[varName] = initValue; // 초기값을 설정
        }
    };

    // 여러 변수들을 초기화하는 함수 정의
    var initVariables = function() {
        initVariable("batch_size", 1); // 배치 크기 초기화
        initVariable("confidence", 0.5); // 신뢰도 초기화
        initVariable("gpu_allocation", 0.5); // GPU 할당 초기화
        initVariable("list_gpu", "0"); // GPU 목록 초기화
        initVariable("record_missing", false); // 누락된 항목 기록 여부 초기화
    };

    // 컨트롤러 초기화 함수 정의
    var init = function() {
        $scope.finishedLoading = false; // 로딩 시작
        initVariables(); // 변수들 초기화
        retrieveCanUseGPU(); // GPU 사용 가능 여부 확인
    };

    // 컨트롤러 초기화 함수 호출
    init();
});
