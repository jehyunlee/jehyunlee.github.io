#!/bin/sh

export CHROME_REMOTE_DESKTOP='/opt/google/chrome-remote-desktop/chrome-remote-desktop'

# 1. stop chrome remote desktop
$CHROME_REMOTE_DESKTOP --stop

# 2. backup the original configuration
cp /opt/google/chrome-remote-desktop/chrome-remote-desktop /opt/google/chrome-remote-desktop/chrome-remote-desktop.orig

# 3. change default size
DEFAULT_SIZES=`grep -r "^DEFAULT_SIZES = .*$" $CHROME_REMOTE_DESKTOP`
echo $DEFAULT_SIZES
sed -i "s/^$DEFAULT_SIZES/#$DEFAULT_SIZES\nDEFAULT_SIZES = \"1920x1080\"/" $CHROME_REMOTE_DESKTOP

# 4. change display number
FIRST_X_DISPLAY_NUMBER=`grep -r "^FIRST_X_DISPLAY_NUMBER = .*$" $CHROME_REMOTE_DESKTOP`
echo $FIRST_X_DISPLAY_NUMBER
sed -i "s/^$FIRST_X_DISPLAY_NUMBER/#$FIRST_X_DISPLAY_NUMBER\nFIRST_X_DISPLAY_NUMBER = 0/" $CHROME_REMOTE_DESKTOP

# 5. comment out sections that look for additional displays
ADD_DISPLAY1=`grep -r "while os.path.exists(X_LOCK_FILE_TEMPLATE % display):" $CHROME_REMOTE_DESKTOP`
ADD_DISPLAY2=`grep -r "display += 1" $CHROME_REMOTE_DESKTOP`
sed -i "s/$ADD_DISPLAY1/#$ADD_DISPLAY1/" $CHROME_REMOTE_DESKTOP
sed -i "s/$ADD_DISPLAY2/#$ADD_DISPLAY2/" $CHROME_REMOTE_DESKTOP

# 6. reuse the existing X session instead of launching a new one.
SELF_LAUNCH_X_SERVER=`grep -r "self._launch_x_server(x_args)" $CHROME_REMOTE_DESKTOP`
SELF_LAUNCH_PRE_SESSION=`grep -r "if not self._launch_pre_session()" $CHROME_REMOTE_DESKTOP`
SELF_LAUNCH_X_SESSION=`grep -r "self.launch_x_session()" $CHROME_REMOTE_DESKTOP`
sed -i "s/$SELF_LAUNCH_X_SERVER/#$SELF_LAUNCH_X_SERVER\n    display=self.get_unused_display_number()\n    self.child_env['DISPLAY']=':%d' % display/" $CHROME_REMOTE_DESKTOP
sed -i "s/$SELF_LAUNCH_PRE_SESSION/#$SELF_LAUNCH_PRE_SESSION/" $CHROME_REMOTE_DESKTOP
sed -i "s/$SELF_LAUNCH_X_SESSION/#$SELF_LAUNCH_X_SESSION/" $CHROME_REMOTE_DESKTOP
