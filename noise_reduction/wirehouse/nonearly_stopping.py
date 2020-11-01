# 训练网络
# total_step = len(train_loader)
# curr_lr = learning_rate

# # 通过epoch循环得到参数
# for epoch in range(num_epochs):
#     for i, (spectrums, labels) in enumerate(train_loader):
#         spectrums = spectrums.to(device)
#         labels = labels.to(device)
#
#         # 输入数据到网络中得到输出
#         outputs = model(spectrums)
#
#         #计算损失
#         loss = criteria(outputs, labels)
#
#         # 反向传播计算参数
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         total_loss.append(loss.item())
#         if (i + 1) % 10 == 0:
#             print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
#                   .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
#
#     # 减少学习率
#     if (epoch + 1) % 20 == 0:
#         curr_lr /= 1.5
#         update_lr(optimizer, curr_lr)
#
# plt.plot(total_loss)
# plt.xlabel('Steps')
# plt.ylabel('Loss')
# plt.show()
#
# # 对网络进行评价
# model.eval()
# model_loss = 0
# test_len = len(test_loader)
# with torch.no_grad():
#     for spectrums, labels in test_loader:
#         spectrums = spectrums.to(device)
#         labels = labels.to(device)
#         outputs = model(spectrums)
#         loss = criteria(outputs, labels)
#         model_loss += loss.item()
#
# avg_loss = model_loss/test_len
# print(avg_loss)
#
# # 保存本次网络的结构参数
# torch.save(model.state_dict(), 'resnet_l2.ckpt')
