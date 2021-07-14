# Unit tests
add_subdirectory(units)

if(NOT MSVC)
  # Testing apps
  set(APP_TESTS
      logger
      dropout
      sqlite
      prod
      cli
      pooling
  )

  if(CUDA_FOUND)
    cuda_add_executable(standalone_test standalone.cu)

    foreach(exec
            standalone_test
            )
      target_link_libraries(${exec} marian ${EXT_LIBS} marian_cuda ${EXT_LIBS})
      cuda_add_cublas_to_target(${exec})
      set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    endforeach(exec)
  endif(CUDA_FOUND)

#  if(CUDNN_FOUND)
#    cuda_add_executable(conv_test conv_test.cu)
#
#    foreach(exec
#            conv_test
#            )
#      target_link_libraries(${exec} marian ${EXT_LIBS})
#      cuda_add_cublas_to_target(${exec})
#      set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
#    endforeach(exec)
#  endif(CUDNN_FOUND)

#  if(CUDA_FOUND)
#    add_executable(standalone_test standalone.cu)
#    target_link_libraries(standalone_test ${EXT_LIBS} marian ${EXT_LIBS} marian_cuda ${EXT_LIBS})
#  endif(CUDA_FOUND)

  foreach(test ${APP_TESTS})
    add_executable("test_${test}" "${test}.cpp")

    if(CUDA_FOUND)
      target_link_libraries("test_${test}" ${EXT_LIBS} marian ${EXT_LIBS} marian_cuda ${EXT_LIBS})
    else(CUDA_FOUND)
      target_link_libraries("test_${test}" marian ${EXT_LIBS})
    endif(CUDA_FOUND)

    set_target_properties("test_${test}" PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  endforeach(test)
endif(NOT MSVC)