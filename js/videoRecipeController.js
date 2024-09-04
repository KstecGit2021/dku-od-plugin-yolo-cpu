// 'detectionRecipe.video'라는 AngularJS 모듈을 생성
var app = angular.module('detectionRecipe.video', []);

// 'videoRecipeController'라는 컨트롤러를 정의하고, $scope를 주입
app.controller('videoRecipeController', function($scope) {

    // GPU 사용 가능 여부와 비디오 관련 정보를 가져오는 함수 정의
    var retrieveCanUseGPU = function() {
        // Python 메소드를 호출하여 비디오 정보와 GPU 사용 가능 여부를 가져옴
        $scope.callPythonDo({method: "get-video-info"}).then(function(data) {
            $scope.canUseGPU = data["can_use_gpu"]; // GPU 사용 가능 여부 설정
            $scope.columns = data['columns']; // 비디오 정보 설정
            $scope.finishedLoading = true; // 로딩 완료 상태로 설정
        }, function(data) {
            $scope.canUseGPU = false; // GPU 사용 불가능으로 설정
            $scope.finishedLoading = true; // 로딩 완료 상태로 설정
        });
    };

    // 특정 변수(varName)가 정의되지 않았을 경우 초기값(initValue)으로 설정하는 함수
    var initVariable = function(varName, initValue) {
        if ($scope.config[varName] == undefined) {
            $scope.config[varName] = initValue; // 변수의 초기값 설정
        }
    };

    // 여러 변수를 초기화하는 함수 정의
    var initVariables = function() {
        initVariable("video_name", "video.mp4"); // 비디오 파일 이름 초기화
        initVariable("detection_rate", 1); // 탐지 비율 초기화
        initVariable("detection_custom", false); // 사용자 정의 탐지 여부 초기화
        initVariable("confidence", 0.5); // 신뢰도 임계값 초기화
        initVariable("gpu_allocation", 0.5); // GPU 할당 비율 초기화
        initVariable("list_gpu", "0"); // 사용할 GPU 목록 초기화
    };

    // 컨트롤러 초기화 함수 정의
    var init = function() {
        $scope.finishedLoading = false; // 로딩 상태 초기화
        initVariables(); // 변수들 초기화
        retrieveCanUseGPU(); // GPU 사용 가능 여부와 비디오 정보 가져오기
    };

    // 컨트롤러 초기화 함수 호출
    init();
});
